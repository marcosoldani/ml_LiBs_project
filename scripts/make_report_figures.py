"""Generate report figures from real benchmarks.

Outputs:
    docs/sources/figures/dataset_nyquist.pdf              — Nyquist a un Aging
    docs/sources/figures/dataset_boxplot_zreal.pdf        — boxplot Re(Z) per Aging
    docs/sources/figures/task1_actual_vs_pred.pdf         — curve previste vs vere
    docs/sources/figures/task2_per_temp_accuracy.pdf      — accuracy per temperatura
    docs/sources/figures/task2_grid_classification.pdf    — 8 panel Nyquist con classe predetta
    docs/sources/figures/task2_feature_importance.pdf     — importance feature per la classificazione
    docs/sources/figures/task3_per_temp.pdf               — R² per temperatura
    docs/sources/figures/task3_grid_sample.pdf            — 2x2 sample previste vs vere
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.models.task1_loo import run_leave_one_out
from src.models.task2_classification import run_classification
from src.models.task3_aging import run_aging_interpolation

OUT = Path("docs/sources/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


# ── Dataset ────────────────────────────────────────────────────────────────
def fig_dataset_nyquist(df: pd.DataFrame) -> None:
    """Nyquist scatter at Aging=2 across all temperatures (SOC=2)."""
    sub = df[(df["Aging"] == 2) & (df["SOC"] == 2)].copy()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    cmap = plt.cm.coolwarm
    temps = sorted(sub["Temperature"].unique())
    for i, t in enumerate(temps):
        rows = sub[sub["Temperature"] == t].sort_values("Frequency", ascending=False)
        ax.plot(
            rows["Z_real"], -rows["Z_imag"],
            "-", marker="o", markersize=2.5, linewidth=1,
            color=cmap(i / max(len(temps) - 1, 1)),
            label=f"{t}°C",
        )
    ax.set_xlabel(r"$\mathrm{Re}(Z)$ [m$\Omega$]")
    ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [m$\Omega$]")
    ax.set_title(r"Curve di Nyquist a Aging$=2$, SOC$=2$")
    ax.grid(alpha=0.3)
    ax.legend(title="Temp.", ncol=2, loc="upper left", fontsize=7)
    fig.savefig(OUT / "dataset_nyquist.pdf")
    plt.close(fig)


def fig_dataset_boxplot(df: pd.DataFrame) -> None:
    """Boxplot di Z_real per Aging (mostra spostamento con invecchiamento)."""
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    data = [df[df["Aging"] == a]["Z_real"].values for a in range(5)]
    bp = ax.boxplot(
        data, labels=[f"Aging {a}" for a in range(5)],
        patch_artist=True, showfliers=False, widths=0.55,
    )
    cmap = plt.cm.viridis
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / 4))
        patch.set_alpha(0.7)
    ax.set_ylabel(r"$\mathrm{Re}(Z)$ [m$\Omega$]")
    ax.set_title(r"Distribuzione di $\mathrm{Re}(Z)$ per livello di invecchiamento")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "dataset_boxplot_zreal.pdf")
    plt.close(fig)


# ── Task 1 ────────────────────────────────────────────────────────────────
def fig_task1(df: pd.DataFrame) -> None:
    res = run_leave_one_out(aging=2, temperature=22.5, df=df)
    df_test = res.df_test.copy()
    df_test["Z_real_pred"] = res.best_predictions[:, 0]
    df_test["Z_imag_pred"] = res.best_predictions[:, 1]

    socs = sorted(df_test["SOC"].unique())
    fig, axes = plt.subplots(1, len(socs), figsize=(2.6 * len(socs), 3.3), sharey=True)
    for ax, soc in zip(axes, socs, strict=True):
        rows = df_test[df_test["SOC"] == soc].sort_values("Frequency", ascending=False)
        ax.plot(rows["Z_real"], -rows["Z_imag"], "o-",
                color="#222", label="Vero", markersize=2.5, linewidth=1)
        ax.plot(rows["Z_real_pred"], -rows["Z_imag_pred"], "x--",
                color="#d62728", label="Predetto", markersize=4, linewidth=1)
        ax.set_title(f"SOC = {soc}")
        ax.set_xlabel(r"$\mathrm{Re}(Z)$ [m$\Omega$]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$-\mathrm{Im}(Z)$ [m$\Omega$]")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        f"Task 1 — Vero vs Predetto (Aging=2, T=22.5°C, modello: {res.best_model_name})",
        fontsize=10, y=1.02,
    )
    fig.savefig(OUT / "task1_actual_vs_pred.pdf")
    plt.close(fig)


# ── Task 2 ────────────────────────────────────────────────────────────────
def fig_task2(df: pd.DataFrame) -> None:
    """Task 2 — held-out (Aging=4, SOC=3): per-temperature accuracy + Nyquist grid."""
    aging_pick, soc_pick = 4, 3
    res = run_classification(aging=aging_pick, soc=soc_pick, df=df)
    best_pred = res.predictions[res.best_model_name]
    pt = res.per_temperature.copy()

    # Per-temperature accuracy bar chart
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = plt.cm.viridis(pt["Accuracy"].values)
    bars = ax.bar(pt["Temperature"].astype(str), pt["Accuracy"],
                  color=colors, alpha=0.9, edgecolor="white")
    ax.set_xlabel("Temperatura [°C]")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Task 2 — accuracy per temperatura "
        f"(escluso Aging={aging_pick}, SOC={soc_pick}, classe vera={res.true_label})"
    )
    for bar, val in zip(bars, pt["Accuracy"].values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "task2_per_temp_accuracy.pdf")
    plt.close(fig)

    # 8-panel grid: predicted class on the held-out Nyquist
    df_test = res.df_test.copy()
    temps = sorted(df_test["Temperature"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6), sharex=False, sharey=False)
    for k, t in enumerate(temps):
        ax = axes[k // 4, k % 4]
        mask = (df_test["Temperature"] == t).values
        sub = df_test.loc[mask].sort_values("Frequency", ascending=False)
        pred_temp = best_pred[mask]
        re_z = sub["Z_real"].values
        im_z = -sub["Z_imag"].values
        for cls, label, color in zip([0, 1], ["Young", "Old"], ["#1b6cf5", "#d62728"], strict=True):
            m = pred_temp == cls
            if m.any():
                ax.scatter(re_z[m], im_z[m], c=color, s=14, alpha=0.85,
                           edgecolors="white", linewidths=0.4, label=f"Pred. {label}")
        n = int(mask.sum())
        n_ok = int((pred_temp == res.true_class).sum())
        ax.set_title(f"T={t}°C — {n_ok}/{n} ok", fontsize=9)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)
    for i in range(2):
        axes[i, 0].set_ylabel(r"$-\mathrm{Im}(Z)$ [m$\Omega$]")
    for j in range(4):
        axes[1, j].set_xlabel(r"$\mathrm{Re}(Z)$ [m$\Omega$]")
    fig.suptitle(
        f"Task 2 — Aging={aging_pick}, SOC={soc_pick} (vero: {res.true_label}) — "
        f"classe predetta dal {res.best_model_name}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT / "task2_grid_classification.pdf")
    plt.close(fig)


# ── Task 3 ────────────────────────────────────────────────────────────────
def fig_task3(df: pd.DataFrame) -> None:
    res = run_aging_interpolation(excluded_aging=2, df=df)
    pt = res.per_temperature.copy()

    # Per-temperature R² + MAE
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].bar(pt["Temperature"].astype(str), pt["R2"],
                color="#1f77b4", alpha=0.85)
    axes[0].set_xlabel("Temperatura [°C]")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_ylim(0.94, 1.0)
    axes[0].set_title(r"$R^2$ per temperatura (KNN, Aging escluso = 2)")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(pt["Temperature"].astype(str), pt["MAE"],
                color="#d62728", alpha=0.85)
    axes[1].set_xlabel("Temperatura [°C]")
    axes[1].set_ylabel(r"MAE [m$\Omega$]")
    axes[1].set_title(r"MAE per temperatura (KNN, Aging escluso = 2)")
    axes[1].grid(alpha=0.3, axis="y")
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    fig.savefig(OUT / "task3_per_temp.pdf")
    plt.close(fig)

    # Sample 2×2 reconstructed curves
    df_test = res.df_test.copy()
    socs_pick = [0, 4]
    temps_pick = [22.5, 40.0]
    fig, axes = plt.subplots(len(temps_pick), len(socs_pick),
                             figsize=(6.4, 5.4), sharex="col")
    for i, t in enumerate(temps_pick):
        for j, soc in enumerate(socs_pick):
            ax = axes[i, j]
            rows = df_test[(df_test["Temperature"] == t) & (df_test["SOC"] == soc)] \
                .sort_values("Frequency", ascending=False)
            ax.plot(rows["Z_real"], -rows["Z_imag"], "o-",
                    color="#222", label="Vero", markersize=2.5, linewidth=1)
            ax.plot(rows["Z_real_pred"], -rows["Z_imag_pred"], "x--",
                    color="#d62728", label="Predetto", markersize=4, linewidth=1)
            ax.set_title(f"T={t}°C, SOC={soc}")
            ax.grid(alpha=0.3)
            if i == len(temps_pick) - 1:
                ax.set_xlabel(r"$\mathrm{Re}(Z)$ [m$\Omega$]")
            if j == 0:
                ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [m$\Omega$]")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Task 3 — Vero vs Predetto, sample di 4 curve",
        fontsize=11, y=1.0,
    )
    fig.savefig(OUT / "task3_grid_sample.pdf")
    plt.close(fig)


def fig_dataset_aging_evolution(df: pd.DataFrame) -> None:
    """Nyquist a (T=25°C, SOC=2) per ogni livello di Aging — mostra l'evoluzione."""
    sub = df[(df["Temperature"] == 25.0) & (df["SOC"] == 2)].copy()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    cmap = plt.cm.viridis
    for a in sorted(sub["Aging"].unique()):
        rows = sub[sub["Aging"] == a].sort_values("Frequency", ascending=False)
        ax.plot(
            rows["Z_real"], -rows["Z_imag"], "-o",
            color=cmap(a / 4), markersize=2.5, linewidth=1.0,
            label=f"Aging {a}",
        )
    ax.set_xlabel(r"$\mathrm{Re}(Z)$ [m$\Omega$]")
    ax.set_ylabel(r"$-\mathrm{Im}(Z)$ [m$\Omega$]")
    ax.set_title(r"Evoluzione delle curve di Nyquist con l'invecchiamento ($T=25\,$°C, SOC$=2$)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "dataset_aging_evolution.pdf")
    plt.close(fig)


def fig_task1_metrics_bar(df: pd.DataFrame) -> None:
    """Bar chart R² per modello (Task 1)."""
    data = json.loads((Path("models/task1") / "benchmark.json").read_text())
    metrics = data["metrics"]
    names = sorted(metrics.keys(), key=lambda n: metrics[n]["R2"], reverse=True)
    r2s = [metrics[n]["R2"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    bars = ax.barh(names[::-1], r2s[::-1], color=colors, alpha=0.85)
    ax.set_xlabel(r"$R^2$ sul fold di test")
    ax.set_xlim(0, 1.05)
    ax.set_title("Task 1 — confronto $R^2$ per modello")
    for bar, val in zip(bars, r2s[::-1], strict=True):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(OUT / "task1_metrics_bar.pdf")
    plt.close(fig)


def fig_task2_feature_importance(df: pd.DataFrame) -> None:
    """Feature importance dal modello migliore (Task 2, escluso Aging=4, SOC=3)."""
    res = run_classification(aging=4, soc=3, df=df)
    fi = res.feature_importance
    if not fi:
        # Fall back to a Random Forest proxy fitted on the same train split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        from src.data.features import build_classification_features
        from src.models.task2_classification import build_labels
        df_l = build_labels(df)
        train = df_l[~((df_l["Aging"] == 4) & (df_l["SOC"] == 3))].reset_index(drop=True)
        X = build_classification_features(train)
        y = train["Age_class"].values
        scaler = StandardScaler()
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(scaler.fit_transform(X), y)
        fi = dict(zip(X.columns, rf.feature_importances_, strict=True))
    items = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(names)))
    ax.barh(names[::-1], vals[::-1], color=colors, alpha=0.85)
    ax.set_xlabel("Importanza relativa")
    ax.set_title("Task 2 — feature importance (Random Forest proxy)")
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(OUT / "task2_feature_importance.pdf")
    plt.close(fig)


def fig_task3_error_distribution(df: pd.DataFrame) -> None:
    """Distribuzione degli errori assoluti (Task 3, KNN, Aging escluso = 2)."""
    res = run_aging_interpolation(excluded_aging=2, df=df)
    err = np.sqrt(
        (res.df_test["Z_real_pred"] - res.df_test["Z_real"]) ** 2
        + (res.df_test["Z_imag_pred"] - res.df_test["Z_imag"]) ** 2
    )
    p95 = float(np.percentile(err, 95))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(err, bins=60, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.axvline(p95, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"95° percentile $= {p95:.3f}$ m$\\Omega$")
    ax.set_xlabel(r"Errore assoluto $\sqrt{\Delta\mathrm{Re}^2 + \Delta\mathrm{Im}^2}$ [m$\Omega$]")
    ax.set_ylabel("Frequenza")
    ax.set_title("Task 3 — distribuzione degli errori sul fold di test")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "task3_error_distribution.pdf")
    plt.close(fig)


def fig_task3_metrics_bar(df: pd.DataFrame) -> None:
    """Bar chart R² per modello (Task 3)."""
    data = json.loads((Path("models/task3") / "benchmark.json").read_text())
    metrics = data["metrics"]
    names = sorted(metrics.keys(), key=lambda n: metrics[n]["R2"], reverse=True)
    r2s = [metrics[n]["R2"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    bars = ax.barh(names[::-1], r2s[::-1], color=colors, alpha=0.85)
    ax.set_xlabel(r"$R^2$ sul fold di test")
    ax.set_xlim(0, 1.05)
    ax.set_title("Task 3 — confronto $R^2$ per modello")
    for bar, val in zip(bars, r2s[::-1], strict=True):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(OUT / "task3_metrics_bar.pdf")
    plt.close(fig)


def fig_task2_cross_pair_heatmap(df: pd.DataFrame) -> None:
    """Heatmap 5×5 dell'accuracy del modello migliore su tutti gli hold-out."""
    import tempfile
    import warnings

    from src.models.task2_classification import benchmark_full_grid

    warnings.filterwarnings("ignore")

    # Riusa la funzione canonica `benchmark_full_grid` per popolare il file
    # JSON, poi lo legge e disegna la heatmap. Niente duplicazione di logica.
    with tempfile.TemporaryDirectory() as tmp:
        out = benchmark_full_grid(out_dir=Path(tmp), df=df)
        full = json.loads(out.read_text())

    grid = np.zeros((5, 5))
    labels = np.empty((5, 5), dtype=object)
    for entry in full.values():
        a, s = int(entry["aging"]), int(entry["soc"])
        acc = float(entry["metrics"][entry["best_model"]]["Accuracy"])
        grid[a, s] = acc
        labels[a, s] = f"{acc:.2f}\n{entry['best_model'].split()[0]}"

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"SOC {s}" for s in range(5)])
    ax.set_yticklabels([f"Aging {a}" for a in range(5)])
    ax.set_title(
        "Task 2 — accuracy del modello migliore per ogni coppia (Aging, SOC)\n"
        "in hold-out (verde = facile, rosso = al confine young/old)",
        fontsize=10,
    )
    for a in range(5):
        for s in range(5):
            ax.text(s, a, labels[a, s], ha="center", va="center",
                    fontsize=7, color="black" if grid[a, s] > 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.savefig(OUT / "task2_cross_pair_heatmap.pdf")
    plt.close(fig)


def fig_task1_residuals_per_soc(df: pd.DataFrame) -> None:
    """Boxplot dei residui assoluti per ciascun SOC (Task 1, hold-out di default)."""
    res = run_leave_one_out(aging=2, temperature=22.5, df=df)
    df_test = res.df_test.copy()
    pred = res.best_predictions
    df_test["err_real"] = pred[:, 0] - df_test["Z_real"].values
    df_test["err_imag"] = pred[:, 1] - df_test["Z_imag"].values
    df_test["err_abs"] = np.sqrt(df_test["err_real"] ** 2 + df_test["err_imag"] ** 2)

    socs = sorted(df_test["SOC"].unique())
    data_by_soc = [df_test[df_test["SOC"] == s]["err_abs"].values for s in socs]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bp = ax.boxplot(data_by_soc, labels=[f"SOC {s}" for s in socs],
                    patch_artist=True, widths=0.55)
    cmap = plt.cm.viridis(np.linspace(0.25, 0.85, len(socs)))
    for patch, c in zip(bp["boxes"], cmap, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.78)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)
    ax.set_ylabel(r"$|\,\mathrm{errore}\,|$ [m$\Omega$]")
    ax.set_title(
        f"Task 1 — distribuzione dei residui per SOC "
        f"(coppia esclusa: Aging=2, $T=22.5$\\,°C, modello: {res.best_model_name})"
    )
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(OUT / "task1_residuals_per_soc.pdf")
    plt.close(fig)


def main() -> None:
    df = load_dataset()
    print("Generating dataset figures...")
    fig_dataset_nyquist(df)
    fig_dataset_aging_evolution(df)
    fig_dataset_boxplot(df)
    print("Generating Task 1 figures...")
    fig_task1(df)
    fig_task1_metrics_bar(df)
    fig_task1_residuals_per_soc(df)
    print("Generating Task 2 figures...")
    fig_task2(df)
    fig_task2_feature_importance(df)
    fig_task2_cross_pair_heatmap(df)
    print("Generating Task 3 figures...")
    fig_task3(df)
    fig_task3_metrics_bar(df)
    fig_task3_error_distribution(df)
    print(f"All figures written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
