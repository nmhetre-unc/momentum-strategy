"""
Where a service measurement ran, recorded with every result: the host
(CPU, power state, Windows power plan), the Docker engine and the CPUs
and memory of the VM it runs in, the image and the git commit.

On this machine Docker Desktop uses the WSL2 backend, so its VM is the
WSL2 utility VM: the CPUs and memory below are shared with the WSL
distro the load generator runs in. With no .wslconfig the VM gets every
logical CPU and half the host's memory.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from machine import _cpu_model, _git_commit, _power_state  # noqa: E402


def _run(*command: str) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def on_ac_power() -> bool:
    return _power_state().get("AC1.online") == "1"


# Windows System-log events that make a measurement invalid: sleep (42),
# wake (Power-Troubleshooter 1), Modern Standby entry/exit (506/507) and a
# change of power source, e.g. AC unplugged (105).
_POWER_EVENT_IDS = (1, 42, 105, 506, 507)


def power_events_since(start: datetime) -> list[dict] | None:
    """Sleep, wake and power-source events since `start` (aware datetime);
    None if the Windows event log can't be read."""
    since = start.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    ids = ",".join(map(str, _POWER_EVENT_IDS))
    script = (f"Get-WinEvent -FilterHashtable @{{LogName='System'; StartTime=[datetime]'{since}'; "
              f"Id={ids}}} -ErrorAction SilentlyContinue | Where-Object {{ $_.Id -ne 1 -or "
              f"$_.ProviderName -eq 'Microsoft-Windows-Power-Troubleshooter' }} | "
              f"ForEach-Object {{ '{{0}} {{1}} {{2}}' -f $_.TimeCreated.ToString('o'), $_.Id, "
              f"$_.ProviderName }}; exit 0")
    # exit 0: Get-WinEvent reports "no events found" as an error, which
    # would otherwise make an empty (good) window look like an unreadable log.
    raw = _run("powershell.exe", "-NoProfile", "-Command", script)
    if raw is None:
        return None
    events = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 3:
            events.append({"time": parts[0], "id": int(parts[1]), "provider": parts[2]})
    return events


class KeepAwake:
    """Asks Windows not to sleep while it exists (SetThreadExecutionState
    with ES_CONTINUOUS | ES_SYSTEM_REQUIRED, held by one powershell.exe).
    Idle sleep only: closing the lid can still suspend the machine."""

    SCRIPT = ("Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] "
              "public static extern uint SetThreadExecutionState(uint f);'; "
              "[W.P]::SetThreadExecutionState([uint32]'0x80000001') | Out-Null; "
              "while ($true) { Start-Sleep -Seconds 60 }")

    def __enter__(self) -> "KeepAwake":
        try:
            self.process: subprocess.Popen | None = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", self.SCRIPT],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.process = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self.process is not None:
            self.process.kill()  # the execution state ends with the process
            self.process.wait()


def docker_engine() -> dict:
    raw = _run("docker", "info", "--format", "{{json .}}")
    if raw is None:
        return {"error": "docker info failed"}
    info = json.loads(raw)
    return {
        "operating_system": info.get("OperatingSystem"),
        "server_version": info.get("ServerVersion"),
        "kernel": info.get("KernelVersion"),
        "vm_cpus": info.get("NCPU"),
        "vm_memory_bytes": info.get("MemTotal"),
        "vm_memory_gib": round(info.get("MemTotal", 0) / 2**30, 2),
        "storage_driver": info.get("Driver"),
        "cgroup_version": info.get("CgroupVersion"),
        "compose_version": _run("docker", "compose", "version", "--short"),
    }


def _windows() -> dict:
    """Host details through powershell.exe (WSL interop); empty elsewhere."""
    script = ("$p = Get-CimInstance Win32_Processor; $c = Get-CimInstance Win32_ComputerSystem; "
              "$b = Get-CimInstance Win32_Battery; "
              "[pscustomobject]@{cores=$p.NumberOfCores; threads=$p.NumberOfLogicalProcessors; "
              "max_mhz=$p.MaxClockSpeed; ram=$c.TotalPhysicalMemory; "
              "battery_status=$b.BatteryStatus; plan=(powercfg /getactivescheme)} "
              "| ConvertTo-Json")
    raw = _run("powershell.exe", "-NoProfile", "-Command", script)
    if not raw:
        return {}
    data = json.loads(raw)
    # Win32_Battery.BatteryStatus: 1 discharging, 2 on AC (also 6-9: charging).
    return {
        "physical_cores": data.get("cores"),
        "logical_processors": data.get("threads"),
        "max_clock_mhz": data.get("max_mhz"),
        "ram_gib": round(int(data.get("ram") or 0) / 2**30, 1),
        "battery_status_code": data.get("battery_status"),
        "power_plan": (data.get("plan") or "").split("(")[-1].rstrip(")").strip(),
    }


def _wslconfig() -> str:
    for home in Path("/mnt/c/Users").glob("*/.wslconfig"):
        return home.read_text()
    return "absent (WSL defaults: all logical CPUs, 50% of host RAM)"


def image_id(image: str = "qbt-service:dev") -> str | None:
    return _run("docker", "image", "inspect", "--format", "{{.Id}}", image)


def environment() -> dict:
    try:
        load = list(os.getloadavg())
    except OSError:
        load = []
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": {
            "cpu_model": _cpu_model(),
            "windows": _windows(),
            "power": _power_state(),
            "on_ac_power": on_ac_power(),
            "wslconfig": _wslconfig(),
        },
        "wsl_distro": {
            "os": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "loadavg_1_5_15": load,
        },
        "docker": docker_engine(),
        "image": image_id(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
    }


if __name__ == "__main__":
    print(json.dumps(environment(), indent=2))
