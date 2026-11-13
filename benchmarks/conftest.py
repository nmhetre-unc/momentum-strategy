import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import machine  # noqa: E402


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info["qbt"] = machine.machine_info()
