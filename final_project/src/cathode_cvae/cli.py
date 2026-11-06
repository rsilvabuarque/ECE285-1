"""One entry point; MLIP dependencies stay in their isolated environments."""

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "train-mask": ("cathode_cvae.learning.train_mask", True),
        "generate-mask": ("cathode_cvae.learning.generate_mask", True),
        "train-geometry": ("cathode_cvae.learning.train_geometry", True),
        "generate-geometry": ("cathode_cvae.learning.generate_geometry", True),
        "evaluate": ("cathode_cvae.evaluation", True),
        "evaluate-geometry": ("cathode_cvae.geometry_evaluation", True),
        "data": ("cathode_cvae.data", False),
        "mlip": ("cathode_cvae.mlip.cli", False),
    }
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=list(commands))
        parser.print_help()
        return 0
    name, *rest = argv
    if name not in commands:
        raise SystemExit(f"Unknown command {name!r}; use --help")
    module, allocation_required = commands[name]
    if allocation_required and not any(x in rest for x in ("--help", "-h")):
        from .runtime import require_compute_allocation
        require_compute_allocation()
    from importlib import import_module
    result = import_module(module).main(rest)
    # Scientific functions can return a receipt for Python callers; console
    # entry points must return an exit code, not hand that dictionary to sys.exit.
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
