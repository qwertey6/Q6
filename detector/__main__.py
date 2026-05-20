"""Allow ``python -m detector`` to invoke the CLI."""
from .core import _main_cli
import sys

raise SystemExit(_main_cli(sys.argv[1:]))
