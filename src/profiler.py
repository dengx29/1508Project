# Track time, GPU use, and search results

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch


class Profiler:
    # This class saves data about how the test runs
    def __init__(self, config: dict | None = None):
        self._process = psutil.Process()
        self._has_cuda = torch.cuda.is_available()
        self._current_stage: str | None = None
        self._stage_start: float | None = None

        # Create the data storage structure
        self.data: dict = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config or {},
                "gpu_device": torch.cuda.get_device_name(0) if self._has_cuda else None,
                "gpu_total_vram_bytes": (
                    torch.cuda.get_device_properties(0).total_memory if self._has_cuda else None
                ),
            },
            "stages": {},
            "queries": [],
            "disk_sizes": {},
        }

    # Start timing a part of the program
    def start_stage(self, name: str) -> None:
        self._current_stage = name
        if self._has_cuda:
            # Reset GPU memory counter to find the peak later
            torch.cuda.reset_peak_memory_stats()
        self._stage_start = time.perf_counter()

    # Stop timing and save memory use
    def end_stage(self, name: str | None = None) -> None:
        stage_name = name or self._current_stage
        if stage_name is None or self._stage_start is None:
            return

        # Calculate time passed
        duration = time.perf_counter() - self._stage_start

        # Record time and memory use
        self.data["stages"][stage_name] = {
            "duration_seconds": round(duration, 4),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated() if self._has_cuda else None
            ),
            "rss_bytes": self._process.memory_info().rss,
        }

        self._current_stage = None
        self._stage_start = None

    # Save results for a single question
    def log_query(self, record: dict) -> None:
        self.data["queries"].append(record)

    # Check how much space a file or folder uses
    def record_disk_size(self, label: str, path: str) -> None:
        p = Path(path)
        if p.is_file():
            total = p.stat().st_size
        elif p.is_dir():
            # Add up sizes of all files in the folder
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        else:
            total = 0
        self.data["disk_sizes"][label] = total

    # Save all collected data to a JSON file
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)