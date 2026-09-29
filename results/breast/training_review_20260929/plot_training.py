"""Rebuild the public diagnostic figure from aggregate JSON files only."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
STAGES = ("representation", "flow", "readout", "joint")
COLORS = ("#243746", "#16816b", "#bb5b33", "#7a5caf")
COMPONENTS = {
    "representation": (
        ("loss", "Weighted training total"),
        ("real_pcr", "Real-trajectory pCR NLL"),
        ("masked_jepa", "Masked JEPA"),
        ("reconstruction", "Reconstruction"),
    ),
    "flow": (
        ("loss", "Weighted training total"),
        ("fm_image", "Image FM"),
        ("fm_state", "State FM"),
    ),
    "readout": (
        ("loss", "Weighted training total"),
        ("marginal_pcr", "Generated-trajectory pCR NLL"),
        ("real_pcr", "Real-trajectory pCR NLL"),
    ),
    "joint": (
        ("loss", "Weighted training total"),
        ("marginal_pcr", "Generated-trajectory pCR NLL"),
        ("fm_image", "Image FM"),
        ("fm_state", "State FM"),
    ),
}


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b9c2c8")
    ax.tick_params(colors="#40505a", labelsize=10)
    ax.grid(axis="y", color="#e3e8eb", linewidth=0.7)
    ax.set_axisbelow(True)


def main():
    summary = json.loads((ROOT / "summary.json").read_text())
    curves = json.loads((ROOT / "training_curves.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(
        4, 2, figsize=(16, 18), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Breast Model: Four-Stage Training Diagnostics",
        x=0.075, y=0.982, ha="left", fontsize=21,
        fontweight="bold", color="#23323d",
    )
    fig.text(
        0.075, 0.959,
        "T0 to T3 run | 764 training patients; 102 development-validation patients | 2026-09-28 to 2026-09-29",
        fontsize=11, color="#52616b",
    )
    for row, stage in enumerate(STAGES):
        info = summary["stages"][stage]
        records = curves["stages"][stage]
        steps = np.asarray([record["step"] for record in records])
        best_step = info["checkpoints"]["best"]["step"]
        last_step = info["checkpoints"]["last"]["step"]
        ax = axes[row, 0]
        style_axis(ax)
        training_max = 0.0
        for color, (key, label) in zip(COLORS, COMPONENTS[stage]):
            values = np.asarray([record["mean"][key] for record in records])
            training_max = max(training_max, float(values.max()))
            ax.plot(steps, values, color=color, linewidth=1.8, marker="o", markersize=2.6, label=label)
        ax.axvline(best_step, color="#6b7378", linestyle=(0, (4, 3)), linewidth=1.3)
        ax.text(
            best_step, 0.99, f"  selected: {best_step:,}",
            transform=ax.get_xaxis_transform(), va="top", fontsize=10, color="#52616b",
        )
        ax.set_title(
            f"{row + 1}. {stage.title()} | sampled training windows",
            loc="left", fontsize=13, fontweight="bold", pad=13,
        )
        ax.set_xlim(0, last_step)
        ax.set_ylim(0, training_max * 1.65)
        ax.set_xlabel("Optimizer step (window end)")
        ax.set_ylabel("Training loss / raw component")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.legend(
            loc="upper right", frameon=True, facecolor="white",
            edgecolor="none", framealpha=1, fontsize=9,
        )

        ax = axes[row, 1]
        style_axis(ax)
        ax.set_title(
            "Saved validation snapshots (NOT a full curve)",
            loc="left", fontsize=12, fontweight="bold", pad=13,
        )
        best, last = info["validation_best"], info["validation_last"]
        labels = [f"Best: step {best_step:,}", f"Last: step {last_step:,}"]
        if stage in {"representation", "flow"}:
            values = [best["selection_score"], last["selection_score"]]
            bars = ax.bar([0, 1], values, width=0.52, color=["#16816b", "#bb5b33"])
            ax.set_xticks([0, 1], labels)
            ax.set_ylabel("Validation selection objective (lower is better)")
            ax.set_ylim(0, max(values) * 1.24)
            ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=5, fontsize=12)
            if stage == "representation":
                components = summary["representation_components"]
                before = components["best"]["component_means"]
                after = components["last"]["component_means"]
                footnote = (
                    "Replayed components (best -> last): reconstruction "
                    f"{before['reconstruction']:.5f} -> {after['reconstruction']:.5f};\n"
                    f"JEPA {before['masked_jepa']:.5f} -> {after['masked_jepa']:.5f}; "
                    f"real pCR NLL {before['real_pcr']:.5f} -> {after['real_pcr']:.5f}."
                )
            else:
                footnote = (
                    "Forward-only FM; 86 paired cases plus 16 zero-loss cases / 102.\n"
                    "Training draws paired patients and also uses reverse intervals."
                )
        else:
            keys = ("nll", "auroc", "auprc", "brier")
            names = ("NLL\n(lower better)", "AUROC\n(higher better)", "AP\n(higher better)", "Brier\n(lower better)")
            x = np.arange(len(keys))
            for offset, metrics, color, label in (
                (-0.19, best, "#16816b", labels[0]),
                (0.19, last, "#bb5b33", labels[1]),
            ):
                values = [metrics[key] for key in keys]
                bars = ax.bar(x + offset, values, width=0.35, color=color, label=label)
                ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=9)
            ax.set_xticks(x, names)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Metric value")
            ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
            footnote = (
                "Selected by minimum NLL, not by maximum AUROC.\n"
                "102 patients; 4 trajectories per patient; 20-step Heun."
                if stage == "readout" else
                "Selected by NLL + 0.05 x generation objective.\n"
                f"Generation objective: {best['generation_objective']:.4f} best; {last['generation_objective']:.4f} last."
            )
        ax.text(
            0, -0.23, footnote, transform=ax.transAxes, fontsize=9,
            va="top", color="#52616b", linespacing=1.5,
        )

    fig.subplots_adjust(left=0.075, right=0.975, top=0.918, bottom=0.17, hspace=0.69, wspace=0.27)
    fig.text(
        0.075, 0.025,
        f"Training points: step 250, every 500 steps, and the selected step; each mean uses up to {curves['window_logged_batches']} preceding logged batches "
        f"(one log / {curves['log_every']} steps).\n"
        "At step 250, only 25 logs are available. Lines connect aggregate windows; they are not the full training history.\n"
        "Validation: only retained best and last snapshots. Loss definitions and sampling differ across stages; compare each stage internally.\n"
        "Development results only, not independent-test results. The generation objective does not measure MRI image quality.",
        fontsize=10, color="#52616b", linespacing=1.7,
    )
    for suffix in ("png", "pdf"):
        path = ROOT / f"training_diagnostics.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path.name)
    plt.close(fig)


if __name__ == "__main__":
    main()
