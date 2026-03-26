#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _paper_x_common import PAPER_ROOT, STAGE_DATA


FIGS = PAPER_ROOT / "figs"
TABLES = PAPER_ROOT / "tables"

COLORS = {
    "exact": "#1f4e79",
    "guarded": "#2a7f62",
    "candidate": "#b45f06",
    "conditional": "#8a3b12",
    "blocked": "#8b1e3f",
    "neutral": "#5f6b7a",
    "light": "#f7f4ee",
    "panel": "#fffdfa",
    "grid": "#d7d2c8",
    "green_fill": "#d9ead3",
    "green_edge": "#7aa86f",
    "green_text": "#305c2f",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def status_color(status: str) -> str:
    status = status.lower()
    if "blocked" in status:
        return COLORS["blocked"]
    if "guarded" in status:
        return COLORS["guarded"]
    if "window" in status:
        return COLORS["exact"]
    if "conditional" in status:
        return COLORS["conditional"]
    if "candidate" in status:
        return COLORS["candidate"]
    return COLORS["neutral"]


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{stem}.{ext}", dpi=240, bbox_inches="tight", facecolor=COLORS["panel"])
    plt.close(fig)


def add_card(ax, x, y, w, h, title, body, color, title_size=11, body_size=9, wrap_width=28) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=color,
        facecolor=COLORS["light"],
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    pad_x = min(0.02, 0.10 * w)
    title_pad_y = min(0.045, 0.18 * h)
    body_pad_y = min(0.12, 0.42 * h)
    ax.text(
        x + pad_x,
        y + h - title_pad_y,
        title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        color=color,
        va="top",
        linespacing=1.15,
    )
    wrapped = []
    for block in body.split("\n"):
        wrapped.extend(textwrap.wrap(block, width=wrap_width) or [""])
    ax.text(
        x + pad_x,
        y + h - body_pad_y,
        "\n".join(wrapped),
        transform=ax.transAxes,
        fontsize=body_size,
        color="#222",
        va="top",
        linespacing=1.28,
    )


def base_figure(width=13, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=COLORS["panel"])
    ax.set_facecolor(COLORS["panel"])
    ax.axis("off")
    return fig, ax


def write_derivation_ladder(rows: list[dict[str, str]]) -> None:
    headers = ["Stage", "Status", "Package summary"]
    table_rows = []
    row_colors = []
    for row in rows:
        summary = "\n".join(
            part.strip() for part in textwrap.wrap(row["claim_safe_summary"].replace("\n", " "), width=108)
        )
        table_rows.append([row["stage"], row["status"], summary])
        row_colors.append(status_color(row["status"]))

    fig, ax = plt.subplots(figsize=(16.0, 5.4), facecolor=COLORS["panel"])
    ax.set_facecolor(COLORS["panel"])
    ax.axis("off")
    ax.text(0.02, 0.96, "Paper X derivation ladder", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.text(
        0.02,
        0.90,
        "A compact stage-by-stage summary of exact, guarded, candidate, and conditional layers.",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["neutral"],
        va="top",
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.09, 0.15, 0.76],
        bbox=[0.02, 0.06, 0.96, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.75)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor(COLORS["grid"])
        if r == 0:
            cell.set_facecolor("#ece7dc")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#222")
        else:
            cell.set_facecolor(COLORS["light"])
            cell.get_text().set_wrap(True)
            if c == 1:
                cell.get_text().set_color(row_colors[r - 1])
                cell.get_text().set_fontweight("bold")
            else:
                cell.get_text().set_color("#222")

    save_figure(fig, "fig01-derivation-ladder")


def write_claim_boundary(rows: list[dict[str, str]]) -> None:
    levels = ["Level_A", "Level_B", "Level_C", "Blocked"]
    counts = [sum(1 for row in rows if row["level"] == level) for level in levels]
    colors = [COLORS["exact"], COLORS["guarded"], COLORS["conditional"], COLORS["blocked"]]

    fig, (ax, ax_note) = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.6),
        facecolor=COLORS["panel"],
        gridspec_kw={"width_ratios": [1.45, 0.9]},
    )
    ax.set_facecolor(COLORS["panel"])
    ax_note.set_facecolor(COLORS["panel"])
    bars = ax.bar(["Level A", "Level B", "Level C", "Blocked"], counts, color=colors, width=0.62)
    ax.set_title("Claim boundary for Paper X", fontsize=18, fontweight="bold", pad=16)
    ax.set_ylabel("Claim count")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, count, color in zip(bars, counts, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.15, str(count), ha="center", va="bottom", fontsize=11, fontweight="bold", color=color)

    ax_note.axis("off")
    add_card(
        ax_note,
        0.08,
        0.20,
        0.84,
        0.56,
        "Interpretation",
        "Level A: exact or guarded core\nLevel B: supported packages\nLevel C: conditional routes\nBlocked: excluded from manuscript claims",
        COLORS["neutral"],
        title_size=11,
        body_size=9,
        wrap_width=28,
    )
    save_figure(fig, "fig02-claim-boundary")


def write_hard_predictions(stage27: dict) -> None:
    pred = stage27["predictions"]
    fig, ax = base_figure(13.5, 4.9)
    ax.text(0.03, 0.93, "Stage 27 hard-prediction package", transform=ax.transAxes, fontsize=18, fontweight="bold")
    ax.text(
        0.03,
        0.86,
        "The electroweak angle remains a matching anchor; the other two remain candidate outputs.",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["neutral"],
    )
    cards = [
        ("$\\Lambda_{\\mathrm{TCV}}$", f'{pred["Lambda_TCV_GeV"]:.3e} GeV\ncandidate calibration\nbenchmark-dependent', COLORS["candidate"]),
        ("$m_{\\Phi_2}$", f'{pred["m_Phi2_eV"]:.1e} eV\ncandidate prediction\nanchor-dependent normalization', COLORS["candidate"]),
        ("$\\sin^2(\\theta_W)$", f'{pred["sin2_theta_w_at_Lambda_TCV"]:.3f} +/- {pred["sin2_theta_w_error"]:.3f}\nexternal matching anchor\nnot internally derived', COLORS["conditional"]),
    ]
    xs = [0.03, 0.355, 0.68]
    for (title, body, color), x in zip(cards, xs):
        add_card(ax, x, 0.14, 0.28, 0.64, title, body, color, wrap_width=26)
    save_figure(fig, "fig03-hard-predictions")


def write_generation_routes(stage28: dict) -> None:
    hs = stage28["headline_structure"]
    fig, ax = base_figure(14.2, 5.1)
    ax.text(0.03, 0.93, "Stage 28 generation-structure routes", transform=ax.transAxes, fontsize=18, fontweight="bold")
    ax.text(
        0.03,
        0.86,
        "All routes remain candidate-level; none is promoted to theorem-level flavour structure.",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["neutral"],
    )
    cards = [
        ("Three-generation route", f'preferred family:\n{hs["preferred_generation_family"]}', COLORS["candidate"]),
        ("CKM-like route", f'triplet balance = {hs["triplet_balance"]:.6f}\nhierarchy margin = {hs["triplet_hierarchy_margin"]:.6f}', COLORS["candidate"]),
        ("Neutrino route", f'neutrino-like role:\n{hs["neutrino_generation_role"]}\ncharged partner:\n{hs["charged_partner_role"]}', COLORS["candidate"]),
    ]
    xs = [0.03, 0.355, 0.68]
    for (title, body, color), x in zip(cards, xs):
        add_card(ax, x, 0.12, 0.28, 0.68, title, body, color, wrap_width=24)
    save_figure(fig, "fig04-generation-routes")


def write_unification_route(stage29: dict) -> None:
    hs = stage29["headline_structure"]
    low_vals = [0.3, 0.65, 1.2]
    labels = [r"$g_1$", r"$g_2$", r"$g_3$"]
    target_lo, target_hi = hs["high_scale_target_window"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.8), facecolor=COLORS["panel"], gridspec_kw={"width_ratios": [1.0, 1.15]})
    for ax in (ax1, ax2):
        ax.set_facecolor(COLORS["light"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.bar(labels, low_vals, color=COLORS["candidate"], width=0.56)
    ax1.set_ylim(0, 1.35)
    ax1.set_title("Low-scale effective family", fontsize=14, fontweight="bold", color=COLORS["candidate"])
    ax1.set_ylabel("Effective coupling")
    ax1.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.text(
        0.04,
        0.93,
        f'family: {hs["low_scale_effective_family"]}\nvariance: {hs["low_scale_variance"]:.6f}',
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
    )

    ax2.set_title("High-scale target near $\\Lambda_{\\mathrm{TCV}}$", fontsize=14, fontweight="bold", color=COLORS["conditional"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.35)
    ax2.set_xticks([])
    ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.axhspan(target_lo, target_hi, color=COLORS["green_fill"], alpha=0.95, label="candidate meeting window")
    ax2.axhline(hs["conditional_sin2_theta_w_anchor"], color=COLORS["conditional"], lw=2.2, label=r"conditional $\sin^2(\theta_W)$ anchor")
    ax2.text(
        0.04,
        0.93,
        f'$\\Lambda_{{\\mathrm{{TCV}}}} = {hs["matching_scale_GeV"]:.3e}\\,\\mathrm{{GeV}}$',
        transform=ax2.transAxes,
        va="top",
        fontsize=9,
    )
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle("Stage 29 conditional unification route", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.05,
        0.02,
        "This figure encodes a candidate meeting route and a conditional 3/8 upgrade path, not a licensed unification claim.",
        fontsize=9,
        color=COLORS["neutral"],
    )
    save_figure(fig, "fig05-unification-route")


def write_sources_note() -> None:
    text = """# Figure Sources

- `fig01-derivation-ladder.(png|pdf)`: `tables/result-summary.csv`
- `fig02-claim-boundary.(png|pdf)`: `tables/claim-hierarchy.csv`
- `fig03-hard-predictions.(png|pdf)`: `papers/sm-stage3-exploration/data/stage27_hard_prediction_assessment.json`
- `fig04-generation-routes.(png|pdf)`: `papers/sm-stage3-exploration/data/stage28_generation_route_assessment.json`
- `fig05-unification-route.(png|pdf)`: `papers/sm-stage3-exploration/data/stage29_unification_route_assessment.json`

These figures are generated from Python with `matplotlib` in the local project virtual environment.
"""
    (FIGS / "FIGURE-SOURCES.md").write_text(text, encoding="utf-8")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    result_rows = load_csv(TABLES / "result-summary.csv")
    claim_rows = load_csv(TABLES / "claim-hierarchy.csv")
    stage27 = load_json(STAGE_DATA / "stage27_hard_prediction_assessment.json")
    stage28 = load_json(STAGE_DATA / "stage28_generation_route_assessment.json")
    stage29 = load_json(STAGE_DATA / "stage29_unification_route_assessment.json")

    write_derivation_ladder(result_rows)
    write_claim_boundary(claim_rows)
    write_hard_predictions(stage27)
    write_generation_routes(stage28)
    write_unification_route(stage29)
    write_sources_note()


if __name__ == "__main__":
    main()
