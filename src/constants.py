"""Shared visualization constants used across modules and UI."""

from __future__ import annotations

TEMP_COLORS: dict[float, str] = {
    20.0: "#1b9e77",
    22.5: "#d95f02",
    25.0: "#7570b3",
    27.5: "#e7298a",
    30.0: "#66a61e",
    35.0: "#e6ab02",
    40.0: "#a6761d",
    47.5: "#1f3a93",
}

AGING_LABELS: dict[int, str] = {
    0: "Aging 0 (Fresh)",
    1: "Aging 1",
    2: "Aging 2 (Mid)",
    3: "Aging 3",
    4: "Aging 4 (Aged)",
}

SOC_LABELS: dict[int, str] = {
    0: "SOC 0 (0%)",
    1: "SOC 1 (25%)",
    2: "SOC 2 (50%)",
    3: "SOC 3 (75%)",
    4: "SOC 4 (100%)",
}

SOC_MARKERS: dict[int, str] = {
    0: "circle",
    1: "square",
    2: "triangle-up",
    3: "diamond",
    4: "triangle-down",
}

CLASS_NAMES: list[str] = ["Young (Aging 0–2)", "Old (Aging 3–4)"]
CLASS_COLORS: list[str] = ["#2ecc71", "#e74c3c"]

AGING_COLORS: dict[int, str] = {
    0: "#3b9df9",
    1: "#27ae60",
    2: "#f39c12",
    3: "#e67e22",
    4: "#c0392b",
}
