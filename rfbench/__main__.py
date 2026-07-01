"""Enable ``python -m rfbench`` to invoke the same entry point as the ``rfbench`` console script."""

from __future__ import annotations

import sys

from rfbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
