"""Render the retained C lead-level metrics; requires optional matplotlib."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from hydra_correctors.baseline_report import read_metrics, validate_bundle


def main():
    directory = root / "results/c_physical_baseline_2019_10"
    validate_bundle(directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "hydra-C-2019-10-v1"})
    rows = read_metrics(directory / "metrics_by_lead.csv")
    leads = [row["lead_hours"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True, layout="constrained")
    for key, label, color in [("RMSE_cms", "RMSE", "#1f4b73"), ("MAE_cms", "MAE", "#b45b26")]:
        axes[0].plot(leads, [row[key] for row in rows], marker="o", markersize=3.5, color=color, label=label)
    axes[0].set(ylabel="Error (m³/s)", ylim=(0, 17))
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(leads, [row["bias_forecast_minus_observation_cms"] for row in rows], marker="o", markersize=3.5, color="#4d7151", label="Forecast − observation bias")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="Forecast lead (hours)", ylabel="Mean bias (m³/s)", ylim=(-1, 8), xticks=[1, 3, 6, 9, 12, 15, 18])
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("C development physical baseline · October 2019\nUSGS 03479000 · 1–18-hour leads", fontsize=13)
    fig.supxlabel("Baseline only; 13,203 scored cells. No paired corrected or learned comparison.", fontsize=9)
    output = directory / "lead_metrics.svg"
    fig.savefig(output, metadata={"Date": None, "Creator": "HYDRA aggregate baseline report"})
    # Canonical line endings keep generated SVG reviewable by git diff --check.
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
