from __future__ import annotations

import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_risk_plot(score: float) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.2))

    color = "#4dd9a6"
    if score >= 75:
        color = "#ff7f7f"
    elif score >= 50:
        color = "#f0d85b"
    elif score >= 25:
        color = "#ffb86b"

    ax.barh(["Risk"], [score], color=color)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk score")
    ax.set_title("Exposure Risk")
    ax.grid(axis="x", alpha=0.25)

    return _fig_to_base64(fig)


def generate_pollution_plot(pm25: float, pm10: float) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.4))

    labels = ["PM2.5", "PM10"]
    values = [pm25, pm10]

    ax.bar(labels, values)
    ax.set_ylabel("µg/m³")
    ax.set_title("Current Pollution")
    ax.grid(axis="y", alpha=0.25)

    return _fig_to_base64(fig)
