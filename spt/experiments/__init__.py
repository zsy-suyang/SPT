"""Reproducible numerical experiments (separate from figure rendering)."""


def main(argv=None):
    """Dispatch a public experiment without importing GPU packages for help."""
    import argparse
    import importlib
    import sys

    commands = {
        "gaussian": "gaussian", "bootstrap": "bootstrap",
        "roundtrip": "roundtrip", "runtime": "runtime",
        "bands": "bands", "energy-windows": "energy_windows",
        "uncertainty": "uncertainty", "calibration": "calibration",
        "benchmark": "benchmark", "transfer": "transfer",
        "transfer-baselines": "transfer_baselines",
        "measurement-controls": "measurement_controls",
    }
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=commands)
    if not values or values[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    selected = parser.parse_args(values[:1]).experiment
    module = importlib.import_module(f"spt.experiments.{commands[selected]}")
    result = module.main(values[1:])
    return result if isinstance(result, int) else 0
