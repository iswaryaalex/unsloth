# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD GPU monitoring via rocm-smi.

Mirrors the nvidia.py module structure so hardware.py can swap backends
based on IS_ROCM. All functions return the same dict shapes as their
nvidia.py counterparts.

rocm-smi ships with every ROCm release and emits a flat card-keyed JSON
dict when called with --json, e.g.:

    {
      "card0": {
        "GPU use (%)": "87",
        "Temperature (Sensor edge) (C)": "57.0",
        "Average Graphics Package Power (W)": "45.013",
        "Max Graphics Package Power (W)": "N/A",
        "VRAM Total Memory (B)": "17163091968",
        "VRAM Total Used Memory (B)": "3087007744"
      },
      "system": { ... }
    }
"""

import json
import re
import subprocess
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

# rocm-smi flags used for all metric queries
_ROCM_SMI_METRIC_FLAGS = [
    "--showuse",
    "--showmeminfo", "vram",
    "--showpower",
    "--showmaxpower",
    "--showtemp",
]


def _run_rocm_smi(*args: str, timeout: int = 5) -> Optional[Any]:
    """Run rocm-smi with the given arguments and return parsed JSON, or None."""
    try:
        result = subprocess.run(
            ["rocm-smi", *args, "--json"],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("rocm-smi query failed: %s", e)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        logger.warning("rocm-smi returned code %d", result.returncode)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Failed to parse rocm-smi JSON output")
        return None


def _parse_numeric(value: Any) -> Optional[float]:
    """Parse a scalar value from rocm-smi output to float, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        import math
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("n/a", "none", "unknown", "disabled", "unsupported"):
            return None
        # Strip trailing unit characters (W, C, %, °, etc.)
        cleaned = re.sub(r"\s*[A-Za-z°/%]+$", "", cleaned)
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None


def _parse_memory_bytes_to_mb(value: Any) -> Optional[float]:
    """Parse a memory value in bytes (as returned by rocm-smi) to MB."""
    num = _parse_numeric(value)
    if num is None:
        return None
    # rocm-smi reports VRAM in bytes
    return num / (1024 * 1024)


def _first_key_match(d: dict, *patterns: str) -> Optional[float]:
    """Return the float value of the first key matching any regex pattern."""
    for pat in patterns:
        for key, val in d.items():
            if re.search(pat, key, re.IGNORECASE):
                return _parse_numeric(val)
    return None


def _extract_gpu_metrics(gpu_data: dict) -> dict[str, Any]:
    """Extract standardized metrics from a single GPU's rocm-smi JSON entry."""
    gpu_util = _first_key_match(gpu_data, r"GPU use\s*\(%\)")
    temp = _first_key_match(
        gpu_data,
        r"Temperature.*edge.*\(C\)",
        r"Temperature.*\(C\)",
    )
    power_draw = _first_key_match(
        gpu_data,
        r"Average Graphics Package Power",
        r"Current Socket Power",
        r"Socket Power",
    )
    power_limit = _first_key_match(
        gpu_data,
        r"Max Graphics Package Power",
        r"PowerCap",
        r"Power Cap",
    )

    # VRAM is reported in bytes by rocm-smi; find by exact key patterns
    vram_total_mb = None
    vram_used_mb = None
    for key, val in gpu_data.items():
        if re.search(r"VRAM Total Memory \(B\)", key, re.IGNORECASE):
            vram_total_mb = _parse_memory_bytes_to_mb(val)
        elif re.search(r"VRAM Total Used Memory \(B\)", key, re.IGNORECASE):
            vram_used_mb = _parse_memory_bytes_to_mb(val)

    vram_used_gb = round(vram_used_mb / 1024, 2) if vram_used_mb is not None else None
    vram_total_gb = round(vram_total_mb / 1024, 2) if vram_total_mb is not None else None
    vram_util = (
        round((vram_used_mb / vram_total_mb) * 100, 1)
        if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0
        else None
    )
    power_util = (
        round((power_draw / power_limit) * 100, 1)
        if power_draw is not None and power_limit and power_limit > 0
        else None
    )

    return {
        "gpu_utilization_pct": gpu_util,
        "temperature_c": temp,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "vram_utilization_pct": vram_util,
        "power_draw_w": power_draw,
        "power_limit_w": power_limit,
        "power_utilization_pct": power_util,
    }


def _card_entries(data: dict) -> list[tuple[int, dict]]:
    """Return sorted (gpu_index, card_data) pairs from a rocm-smi JSON dict.

    rocm-smi keys GPU entries as "card0", "card1", etc. The "system" key
    and any non-card keys are ignored.
    """
    entries = []
    for key, val in data.items():
        m = re.fullmatch(r"card(\d+)", key, re.IGNORECASE)
        if m and isinstance(val, dict):
            entries.append((int(m.group(1)), val))
    entries.sort(key=lambda t: t[0])
    return entries


def get_physical_gpu_count() -> Optional[int]:
    """Return physical AMD GPU count via rocm-smi, or None on failure."""
    data = _run_rocm_smi(*_ROCM_SMI_METRIC_FLAGS)
    if not isinstance(data, dict):
        return None
    count = len(_card_entries(data))
    return count if count > 0 else None


def get_primary_gpu_utilization() -> dict[str, Any]:
    """Return utilization metrics for the primary AMD GPU."""
    data = _run_rocm_smi(*_ROCM_SMI_METRIC_FLAGS)
    if not isinstance(data, dict):
        return {"available": False}

    entries = _card_entries(data)
    if not entries:
        return {"available": False}

    _, gpu_data = entries[0]
    metrics = _extract_gpu_metrics(gpu_data)
    metrics["available"] = True
    return metrics


def get_visible_gpu_utilization(
    parent_visible_ids: Optional[list[int]],
    parent_cuda_visible_devices: Optional[str] = None,
) -> dict[str, Any]:
    """Return utilization metrics for visible AMD GPUs."""
    if parent_visible_ids is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": [],
            "devices": [],
            "index_kind": "unresolved",
        }

    data = _run_rocm_smi(*_ROCM_SMI_METRIC_FLAGS)
    if not isinstance(data, dict):
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }

    visible_set = set(parent_visible_ids)
    ordinal_map = {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}

    devices = []
    for idx, gpu_data in _card_entries(data):
        if idx not in visible_set:
            continue
        metrics = _extract_gpu_metrics(gpu_data)
        metrics["index"] = idx
        metrics["index_kind"] = "physical"
        metrics["visible_ordinal"] = ordinal_map.get(idx, len(devices))
        devices.append(metrics)

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": parent_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
    }
