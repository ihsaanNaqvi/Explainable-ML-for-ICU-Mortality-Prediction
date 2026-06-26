"""
Pre-generate clean assets for the defence deck — v3.
Every image is built at the exact aspect ratio of its embed slot with
generous padding so nothing is cut or overlaps.

Assets produced (in d:/icu-xai/outputs/figures/deck_assets/):
  title_ecg_strip.png    : 9.0" x 1.6" clean ECG strip for the title slide
  novelty_visual.png     : 12.5" x 4.8" three-pillar diagram, headers FIT
  qr_github.png          : QR code for the GitHub repo
  qr_physionet.png       : QR code for the PhysioNet 2012 dataset
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import qrcode
from qrcode.constants import ERROR_CORRECT_M

ASSETS = Path(r"d:/icu-xai/outputs/figures/deck_assets")
ASSETS.mkdir(parents=True, exist_ok=True)

TSU_BLUE_DARK = "#0B2F55"
TSU_BLUE      = "#1F6AB4"
TSU_BLUE_LT   = "#D6E4F0"
WHITE         = "#FFFFFF"
ECG_CYAN      = "#7FD3F7"


# ============================================================
# 1. TITLE ECG STRIP  —  9.0 x 1.6 in (aspect 5.6:1)
# ============================================================
def make_title_ecg_strip():
    W, H = 9.0, 1.6
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.set_facecolor(TSU_BLUE_DARK)
    fig.patch.set_facecolor(TSU_BLUE_DARK)
    ax.set_axis_off()

    # Faint monitor gridlines
    for yh in (8, 15, 22):
        ax.axhline(yh, color="white", alpha=0.06, linewidth=0.6)

    # ECG beats
    x = np.linspace(2, 98, 3000)
    y = np.full_like(x, 15.0)

    def beat(c):
        y_  =  1.5 * np.exp(-((x - (c - 4.5)) / 1.6) ** 2)
        y_ += -1.5 * np.exp(-((x - (c - 0.6)) / 0.30) ** 2)
        y_ += 10.0 * np.exp(-((x - c) / 0.20) ** 2)
        y_ += -2.5 * np.exp(-((x - (c + 0.6)) / 0.30) ** 2)
        y_ +=  3.0 * np.exp(-((x - (c + 5.0)) / 1.8) ** 2)
        return y_

    for c in (12, 32, 52, 72, 92):
        y = y + beat(c)

    ax.plot(x, y, color=ECG_CYAN, linewidth=8.0, alpha=0.15)  # glow
    ax.plot(x, y, color=ECG_CYAN, linewidth=2.2, alpha=1.0)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = ASSETS / "title_ecg_strip.png"
    fig.savefig(out, dpi=240, facecolor=TSU_BLUE_DARK,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  [ok] {out.name}  ({W}x{H} in)")


# ============================================================
# 2. NOVELTY VISUAL  —  12.5 x 4.8 in  (aspect 2.60:1)
#    Headers wrapped to 2 lines so nothing is cut.
# ============================================================
def make_novelty_visual():
    W, H = 12.5, 4.8
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 48)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    pillars = [
        dict(
            x=4, color=TSU_BLUE, icon="T",
            title="Time-Aware\nTransformer",
            subtitle="Architecture",
            bullets=[
                "(48 × 74) input with masks",
                "Learnable sinusoidal time PE",
                "CLS-token head, 205k params",
                "AUROC 0.854  ·  Score1 0.530",
            ],
        ),
        dict(
            x=46, color="#0A8754", icon="Φ",
            title="Time-Step SHAP\nAttribution",
            subtitle="Explanation",
            bullets=[
                "TreeSHAP on 227 flat features",
                "Redistribute Φ across 48 hrs",
                "weighted by |X[h, v]|",
                "→ (hour × variable) tensor",
            ],
        ),
        dict(
            x=88, color="#C0392B", icon="✓",
            title="Clinical-Score\nConcordance Test",
            subtitle="Validation",
            bullets=[
                "Per-patient C(p) vs SAPS-I",
                "Mean concordance 0.498",
                "First on PhysioNet 2012",
                "ρ = +0.139  ·  p = 6.3 × 10⁻⁴",
            ],
        ),
    ]

    PW = 33    # pillar width  (was 32 — slightly wider so headers fit)
    PH = 42    # pillar height
    HEADER_H = 12   # taller header → 2-line title fits

    for p in pillars:
        # Card body
        ax.add_patch(mpatches.FancyBboxPatch(
            (p["x"], 2), PW, PH,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            linewidth=2.0, edgecolor=p["color"], facecolor="white"))
        # Header strip (taller, 2-line space)
        ax.add_patch(mpatches.FancyBboxPatch(
            (p["x"], 2 + PH - HEADER_H), PW, HEADER_H + 0.3,
            boxstyle="round,pad=0,rounding_size=1.4",
            linewidth=0, facecolor=p["color"]))
        # Icon disc — centred vertically in the header
        header_mid = 2 + PH - HEADER_H / 2
        ax.add_patch(mpatches.Circle((p["x"] + 4.0, header_mid),
                                     2.6, facecolor="white", edgecolor="none"))
        ax.text(p["x"] + 4.0, header_mid, p["icon"],
                color=p["color"], fontsize=22, fontweight="bold",
                ha="center", va="center")
        # Title (2 lines) — generous left padding past the icon
        ax.text(p["x"] + 8.5, header_mid + 1.6, p["title"],
                color="white", fontsize=13.5, fontweight="bold",
                ha="left", va="center", linespacing=0.95)
        ax.text(p["x"] + 8.5, header_mid - 4.3, p["subtitle"],
                color="#E6EEF7", fontsize=10, ha="left", va="center")
        # Bullets
        y_start = (2 + PH - HEADER_H) - 4.0
        for i, b in enumerate(p["bullets"]):
            yy = y_start - i * 6.0
            ax.text(p["x"] + 2.5, yy, "●",
                    color=p["color"], fontsize=11, va="center")
            ax.text(p["x"] + 5.0, yy, b,
                    color="#1a1a1a", fontsize=10.8, va="center")

    # Arrows between pillars (sit at vertical centre of cards)
    arrow_y = 2 + PH / 2
    for i in range(2):
        x1 = pillars[i]["x"] + PW
        x2 = pillars[i + 1]["x"]
        ax.annotate("", xy=(x2 - 0.6, arrow_y), xytext=(x1 + 0.6, arrow_y),
                    arrowprops=dict(arrowstyle="->,head_length=0.7,head_width=0.45",
                                    color="#888", lw=1.8))

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out = ASSETS / "novelty_visual.png"
    fig.savefig(out, dpi=240, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [ok] {out.name}  ({W}x{H} in)")


# ============================================================
# 3. QR codes
# ============================================================
def make_qr(text, name, fill_color):
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M,
                       box_size=14, border=2)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color=fill_color, back_color="white").convert("RGB")
    out = ASSETS / name
    img.save(out, dpi=(300, 300))
    print(f"  [ok] {name}  ({img.size[0]}x{img.size[1]})  -> {text}")


# ============================================================
if __name__ == "__main__":
    print("Generating deck assets ...")
    make_title_ecg_strip()
    make_novelty_visual()
    make_qr("https://github.com/ihsaanNaqvi/Explainable-ML-for-ICU-Mortality-Prediction",
            "qr_github.png", fill_color=TSU_BLUE_DARK)
    make_qr("https://physionet.org/content/challenge-2012/1.0.0/",
            "qr_physionet.png", fill_color="#0A8754")
    print(f"\nAll assets in {ASSETS}")
