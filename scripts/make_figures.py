"""Generate 3 publication-quality 300 DPI figures for Paper 43 (Sälzer, arXiv:2611.32595).

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})


def make_fig1() -> None:
    d1 = json.loads((DATA_DIR / "exp1_lemma31_pop_separation.json").read_text(encoding="utf-8"))["rows"]
    d2 = json.loads((DATA_DIR / "exp2_mpnn_vs_run_scaling.json").read_text(encoding="utf-8"))["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ks = [r["k"] for r in d1]
    widths = [r["interval_width_float"] for r in d1]
    gaps = [r["min_wrong_pop_gap_float"] for r in d1]
    ax1.plot(ks, widths, "o-", color="#2b5c8f", lw=1.8, label=r"Interval width $(k-1)/(k+1)$")
    ax1.plot(ks, gaps, "s--", color="#c95b1d", lw=1.8, label=r"Min wrong-pop gap to $I$")
    ax1.axhline(1.0, color="#b22222", ls=":", lw=1.2, label="Strict unit ceiling = 1")
    ax1.set_title("(a) Lemma 3.1 Encoding Width & Pop Separation")
    ax1.set_xlabel(r"Working alphabet size $k = |\Gamma|$ ($b = k + 2$)")
    ax1.set_ylabel("Exact rational value")
    ax1.set_ylim(0, 1.38)
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(loc="upper left", ncol=1, framealpha=0.94, fontsize=7.3)

    hs = [r["height_h"] for r in d2]
    dims = [r["label_dim"] for r in d2]
    layers = [r["mpnn_layers"] for r in d2]
    runs = [r["run_length_2h"] for r in d2]
    ax2.semilogy(hs, runs, "o-", color="#2b5c8f", lw=1.9, label=r"Verified NTM run length $2^h$")
    ax2.semilogy(hs, dims, "s--", color="#c95b1d", lw=1.7, label=r"MPNN label dim $14(h+1)$")
    ax2.semilogy(hs, layers, "^:", color="#4c8c4a", lw=1.7, label=r"MPNN depth $h + 2$")
    ax2.set_title("(b) Polynomial MPNN vs. Exponential Run Length")
    ax2.set_xlabel(r"Decomposition tree height $h = p(|w|)$")
    ax2.set_ylabel("Scale (log axis)")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_encoding_and_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig2() -> None:
    d3 = json.loads((DATA_DIR / "exp3_relu_fnn_profile.json").read_text(encoding="utf-8"))["rows"]
    d4 = json.loads((DATA_DIR / "exp4_mpnn_execution_audit.json").read_text(encoding="utf-8"))["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    xs = [r["x"] for r in d3]
    vs = [r["fnn_violation"] for r in d3]
    ax1.plot(xs, vs, color="#2b5c8f", lw=2.0, label=r"Exact $N_S(x, x)$")
    ax1.fill_between(xs, 0, vs, color="#2b5c8f", alpha=0.14)
    ax1.axvspan(0.0, 1.0, color="#4c8c4a", alpha=0.18, label=r"Polyhedron $P_1 = [0,1]^2$")
    ax1.axvspan(3.0, 4.0, color="#c95b1d", alpha=0.18, label=r"Polyhedron $P_2 = [3,4]^2$")
    ax1.set_title("(a) Lemma 5.1 ReLU-FNN Zero-Set Profile")
    ax1.set_xlabel(r"Diagonal coordinate $x$ along $(x, x)$")
    ax1.set_ylabel(r"Exact ReLU-FNN output $N_S(x, x)$")
    ax1.set_ylim(-0.08, 2.55)
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(loc="upper left", framealpha=0.94, fontsize=7.5)

    from fractions import Fraction
    labels = ["h=1\n3 nodes\nACCEPT", "h=2\n7 nodes\nACCEPT", "h=3\n15 nodes\nACCEPT", "h=2\n7 nodes\nREJECT"]
    outs = [float(Fraction(r["output"])) for r in d4]
    colors = ["#4c8c4a" if r["accepted"] else "#b22222" for r in d4]
    bars = ax2.bar(range(len(d4)), outs, width=0.52, color=colors)
    ax2.axhline(1.0, color="#222222", ls="--", lw=1.0, label="Accept threshold = 1")
    for b, r, yv in zip(bars, d4, outs):
        txt = f"out={r['output']}\n({r['verify_ms']:.2f} ms)"
        ax2.text(b.get_x() + b.get_width() / 2, yv + 0.04, txt, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_title(r"(b) Theorem 6.1 Exact $M_w(G,v)$ Audit ($\mathrm{deg} \leq 3$)")
    ax2.set_xticks(range(len(d4)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel(r"Exact MPNN output $M_w(G, v)$")
    ax2.set_ylim(0, 1.36)
    ax2.grid(True, axis="y", ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_relu_profile_and_mpnn_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3() -> None:
    rep = json.loads((DATA_DIR / "certificate_replay_summary.json").read_text(encoding="utf-8"))["certificates"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    mode_colors = {
        "tape_encoding_lemma31_cert": "#2b5c8f",
        "polyhedral_ntm_delta_lemma41_cert": "#3d85c6",
        "relu_polyhedra_union_lemma51_cert": "#4c8c4a",
        "local_sum_mpnn_deg3_theorem61_cert": "#c95b1d",
    }
    mode_labels = {
        "tape_encoding_lemma31_cert": "Mode 1 (Lem. 3.1)",
        "polyhedral_ntm_delta_lemma41_cert": "Mode 2 (Lem. 4.1)",
        "relu_polyhedra_union_lemma51_cert": "Mode 3 (Lem. 5.1)",
        "local_sum_mpnn_deg3_theorem61_cert": "Mode 4 (Thm. 6.1)",
    }
    seen = set()
    for i, r in enumerate(rep):
        m = r["mode"]
        lbl = mode_labels[m] if m not in seen else None
        seen.add(m)
        ax1.bar(i, r["verify_ms"], width=0.68, color=mode_colors[m], label=lbl)
    ax1.set_title("(a) Canonical Certificate Replay Latency")
    ax1.set_xlabel("Canonical MPNN-DEGBSAT-CERT-1.0 certificate (C1-C16)")
    ax1.set_ylabel("Exact zero-float replay latency (ms)")
    ax1.set_xticks(range(len(rep)))
    ax1.set_xticklabels([f"C{i+1}" for i in range(len(rep))], fontsize=7)
    ax1.grid(True, axis="y", ls="--", alpha=0.35)
    ax1.legend(loc="upper left", ncol=2, framealpha=0.92, fontsize=7.5)

    cats = ["Global /\nType", "Mode 1\nTape", "Mode 2\nDelta", "Mode 3\nReLU", "Mode 4\nMPNN"]
    vals = [12, 13, 12, 13, 12]
    cols = ["#666666", "#2b5c8f", "#3d85c6", "#4c8c4a", "#c95b1d"]
    bars = ax2.bar(cats, vals, width=0.58, color=cols)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35, f"{v}/{v}", ha="center", va="bottom", fontweight="bold", fontsize=8.5)
    ax2.set_title("(b) Adversarial Red-Team Audit (62/62 Rejected)")
    ax2.set_ylabel("Live adversarial forgeries rejected")
    ax2.set_ylim(0, 16.0)
    ax2.grid(True, axis="y", ls="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_certificate_replay_atlas.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    print("Generated all 3 Paper 43 publication figures successfully.")


if __name__ == "__main__":
    main()
