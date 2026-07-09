from __future__ import annotations

from run_sparseworld_swvis4_main import build_support_keyframes
from swvis4_support_interp_lib import ensure_dirs


if __name__ == "__main__":
    ensure_dirs()
    build_support_keyframes(3)
