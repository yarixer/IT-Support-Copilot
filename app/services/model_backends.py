from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder:
    def embed_documents(self, texts: Sequence[str], instruction: str | None = None) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str, instruction: str | None = None) -> np.ndarray:
        return self.embed_documents([text], instruction=instruction)[0]

    def warmup(self) -> None:
        _ = self.embed_query("warmup")


class TransformersEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        batch_size: int = 4,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if torch_dtype == "float16" else None
        model_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs)
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.dim = None

    def _mean_pool(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    def embed_documents(self, texts: Sequence[str], instruction: str | None = None) -> np.ndarray:
        if hasattr(self.model, "encode"):
            arrays: list[np.ndarray] = []
            for start in range(0, len(texts), self.batch_size):
                chunk = list(texts[start : start + self.batch_size])
                kwargs = {"max_length": self.max_length}
                if instruction:
                    kwargs["instruction"] = instruction
                embeddings = self.model.encode(chunk, **kwargs)
                arrays.append(np.asarray(embeddings, dtype=np.float32))
                if self.device != "cpu" and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            array = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 0), dtype=np.float32)
        else:
            arrays: list[np.ndarray] = []
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                batch_texts = [f"{instruction}\n{text}" if instruction else text for text in chunk]
                tokens = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                if self.device != "cpu":
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with self.torch.no_grad():
                    outputs = self.model(**tokens)
                if hasattr(outputs, "sentence_embeddings"):
                    pooled = outputs.sentence_embeddings
                else:
                    pooled = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
                arrays.append(pooled.detach().cpu().numpy().astype(np.float32))
                del outputs, pooled, tokens
                if self.device != "cpu" and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            array = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, 0), dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array = array / norms
        self.dim = array.shape[1] if array.size else 0
        return array


class TransformersReranker:
    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 4, max_length: int = 768):
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        self.cross_encoder = None
        self.seq_cls = None
        self.tokenizer = None

        try:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(
                model_id,
                trust_remote_code=True,
                device=device,
            )
            logger.info("Loaded reranker via sentence-transformers CrossEncoder.")
            return
        except Exception as exc:
            logger.warning("CrossEncoder load failed, falling back to transformers: %s", exc)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.seq_cls = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
        if device != "cpu":
            self.seq_cls = self.seq_cls.to(device)
        self.seq_cls.eval()

    def warmup(self) -> None:
        _ = self.score("warmup query", ["warmup candidate"])

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        if self.cross_encoder is not None:
            pairs = [[query, doc] for doc in documents]
            scores = self.cross_encoder.predict(
                pairs,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            return [float(x) for x in scores]

        import torch

        scores: list[float] = []
        pairs = [[query, doc] for doc in documents]
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device != "cpu":
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = self.seq_cls(**tokens)
            logits = outputs.logits.squeeze(-1).detach().cpu().numpy()
            scores.extend(float(x) for x in np.atleast_1d(logits))
            del outputs, tokens
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores


class TransformersGenerator:
    def __init__(self, model_id: str, device: str = "cpu", torch_dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16

        model_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device != "cpu":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if device == "cpu":
            self.model = self.model.to(device)
        if hasattr(self.model, "generation_config"):
            for attr in ["top_k", "min_p", "top_h", "temperature_last"]:
                if hasattr(self.model.generation_config, attr):
                    try:
                        setattr(self.model.generation_config, attr, None)
                    except Exception:
                        pass
        self.model.eval()

    def warmup(self) -> None:
        _ = self.generate("Return JSON with answer warmup.", max_new_tokens=8, temperature=0.0, top_p=1.0)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 320,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a precise payment-support analyst. Use only grounded evidence from the retrieved cases."},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=bool(temperature and temperature > 0),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if temperature and temperature > 0:
            generate_kwargs["temperature"] = float(temperature)
            generate_kwargs["top_p"] = float(top_p)
        with self.torch.no_grad():
            output = self.model.generate(**generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
