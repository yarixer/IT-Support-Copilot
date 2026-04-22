from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime import preload_runtime, get_runtime_status

if __name__ == "__main__":
    preload_runtime()
    print(get_runtime_status())
