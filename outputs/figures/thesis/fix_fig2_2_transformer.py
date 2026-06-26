"""
Fix Figure 2.2 — Time-Aware Transformer Architecture schematic.

Standalone matplotlib diagram — no external diagram libraries needed.
Usage (from d:/icu-xai/):
    python outputs/figures/thesis/fix_fig2_2_transformer.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "outputs" / "figures" / "thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────
C_INPUT  = "#2C3E50"
C_PROJ   = "#1565C0"
C_ATTN   = "#00897B"
C_FFN    = "#00695C"
C_MLP    = "#F57F17"
C_CLS    = "#E53935"
C_NORM   = "#546E7A"
C_OUT    = "#2E7D32"
C_ARROW  = "#444444"
WHITE    = "white"
DARK     = "#1a1a1a"

# ── Figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 18), dpi=300)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis("off")

CX = 7.0   # centre x — all main blocks share this x
BW = 8.2   # main-block width
BH = 0.65  # standard block height


# ── Helpers ─────────────────────────────────────────────────────────────
def block(xc, yc, w, h, color, label, sub=None,
          fs=10, tc=WHITE, lw=1.5, ec=WHITE):
    """Rounded rectangle centred at (xc, yc), optional second line."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.12", linewidth=lw,
        edgecolor=ec, facecolor=color, zorder=3))
    if sub:
        ax.text(xc, yc + 0.14, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=4)
        ax.text(xc, yc - 0.17, sub,
                ha="center", va="center", fontsize=fs - 1.5,
                color=tc, alpha=0.92, zorder=4)
    else:
        ax.text(xc, yc, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=4)


def arrow(x, y0, y1, ann=None, ax_off=2.3):
    """Downward arrow from y0 to y1 with optional dimension annotation."""
    ax.annotate(
        "", xy=(x, y1 + 0.04), xytext=(x, y0 - 0.04),
        arrowprops=dict(arrowstyle="->", color=C_ARROW,
                        lw=1.6, mutation_scale=14),
        zorder=2)
    if ann:
        ax.text(x + ax_off, (y0 + y1) / 2, ann,
                fontsize=8.5, color="#666666",
                va="center", style="italic")


# ════════════════════════════════════════════════════════════════════════
# Title
ax.text(CX, 17.62, "Figure 2.2 — Time-Aware Transformer Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=DARK)

# ── 1. INPUT ─────────────────────────────────────────────────────────────
block(CX, 16.88, BW, BH, C_INPUT,
      "Input Tensor",
      sub="shape  (48 × 74)   —   37 normalised values  +  37 missingness masks",
      fs=10)
arrow(CX, 16.55, 16.12, "(48×74)")

# ── 2. Linear Projection ──────────────────────────────────────────────────
block(CX, 15.82, BW, BH, C_PROJ,
      "Linear Projection",
      sub="shared weight matrix applied at every time step   |   74  →  d_model = 64",
      fs=10)
arrow(CX, 15.49, 15.06, "(48×64)")

# ── 3. TAPE ───────────────────────────────────────────────────────────────
block(CX, 14.75, BW, BH + 0.10, C_PROJ,
      "Time-Aware Positional Encoding  (TAPE)",
      sub="PE(pos, 2i) = sin(pos / ωᵢ )   ·   ωᵢ ∈ ℝ^(d/2) learnable, initialised at 10000^(2i/d)",
      fs=9.8)
arrow(CX, 14.39, 13.97, "(48×64) + PE")

# ── 4. CLS Token ─────────────────────────────────────────────────────────
block(CX, 13.67, BW, BH, C_CLS,
      "[CLS] Token Prepended",
      sub="learnable vector prepended at position 0   →   sequence length  48 + 1 = 49",
      fs=10)
arrow(CX, 13.34, 12.97, "(49×64)")

# ════════════════════════════════════════════════════════════════════════
# Transformer Encoder Layer  (expanded, ×4)
TR_TOP = 12.88
TR_BOT = 9.05
TR_W   = 10.2

ax.add_patch(mpatches.FancyBboxPatch(
    (CX - TR_W / 2, TR_BOT), TR_W, TR_TOP - TR_BOT,
    boxstyle="round,pad=0.15", linewidth=2.2,
    edgecolor="#90CAF9", facecolor="#EBF5FB", zorder=1))

# Header
ax.text(CX, TR_TOP - 0.22,
        "Transformer Encoder Layer",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color="#0D47A1")

# ×4 badge (right of box)
bx = CX + TR_W / 2 + 0.60
by = (TR_TOP + TR_BOT) / 2
ax.add_patch(mpatches.FancyBboxPatch(
    (bx - 0.40, by - 0.32), 0.80, 0.64,
    boxstyle="round,pad=0.08", linewidth=1.8,
    edgecolor="#0D47A1", facecolor="#0D47A1", zorder=5))
ax.text(bx, by, "×4", ha="center", va="center",
        fontsize=11, fontweight="bold", color=WHITE, zorder=6)
# Bracket lines from badge to outer box edges
ax.plot([CX + TR_W / 2, bx - 0.40], [TR_TOP, TR_TOP],
        color="#90CAF9", lw=1.6, zorder=2)
ax.plot([CX + TR_W / 2, bx - 0.40], [TR_BOT, TR_BOT],
        color="#90CAF9", lw=1.6, zorder=2)

# Pre-norm annotation
ax.text(CX, 12.48,
        "(pre-LayerNorm applied before each sub-layer,  dropout = 0.1)",
        ha="center", va="center", fontsize=8.5,
        color="#555555", style="italic")

# Inner blocks
IW  = 7.2    # inner block width
IH  = 0.57   # inner block height
IHN = 0.42   # Add&Norm height

MHA_Y = 12.00
AN1_Y = 11.27
FFN_Y = 10.48
AN2_Y = 9.75

block(CX, MHA_Y, IW, IH, C_ATTN,
      "Multi-Head Self-Attention",
      sub="h = 4 heads   ·   d_head = 16   ·   scaled dot-product attention",
      fs=9.5)

block(CX, AN1_Y, IW, IHN, C_NORM,
      "Add  &  LayerNorm   (residual + normalisation)",
      fs=9.2)

block(CX, FFN_Y, IW, IH, C_FFN,
      "Feed-Forward Network",
      sub="Linear 64→256  (ReLU)  →  Linear 256→64   (position-wise, shared weights)",
      fs=9.5)

block(CX, AN2_Y, IW, IHN, C_NORM,
      "Add  &  LayerNorm   (residual + normalisation)",
      fs=9.2)

# Inner flow arrows
for ys, ye in [
    (MHA_Y - IH / 2,   AN1_Y + IHN / 2),
    (AN1_Y - IHN / 2,  FFN_Y + IH / 2),
    (FFN_Y - IH / 2,   AN2_Y + IHN / 2),
]:
    ax.annotate("", xy=(CX, ye + 0.025), xytext=(CX, ys - 0.025),
                arrowprops=dict(arrowstyle="->", color="#999",
                                lw=1.15, mutation_scale=11), zorder=4)

# Residual skip lines (right side of inner blocks)
RX = CX + IW / 2 + 0.52
for y_top, y_bot in [
    (MHA_Y + IH / 2,   AN1_Y + IHN / 2),  # bypass MHA
    (AN1_Y + IHN / 2,  AN2_Y + IHN / 2),  # bypass FFN
]:
    ax.plot([CX + IW / 2, RX, RX, CX + IW / 2],
            [y_top, y_top, y_bot, y_bot],
            color="#90A4AE", lw=1.25, zorder=2)
    ax.annotate("", xy=(CX + IW / 2, y_bot), xytext=(RX, y_bot),
                arrowprops=dict(arrowstyle="->", color="#90A4AE",
                                lw=1.25, mutation_scale=10), zorder=4)

# ════════════════════════════════════════════════════════════════════════
arrow(CX, TR_BOT, 8.65, "(49×64)")

# ── 5. CLS extracted ──────────────────────────────────────────────────────
block(CX, 8.32, BW, BH, C_CLS,
      "CLS Vector Extracted",
      sub="row 0 of encoder output  —  contextualised sequence representation   shape (64,)",
      fs=10)
arrow(CX, 7.99, 7.56, "(64,)")

# ── 6. MLP Dense 1 ───────────────────────────────────────────────────────
block(CX, 7.22, BW, BH, C_MLP,
      "MLP Head — Dense Layer 1",
      sub="Linear  64  →  32   |   GELU activation   |   dropout 0.1",
      fs=10)
arrow(CX, 6.89, 6.46, "(32,)")

# ── 7. MLP Dense 2 ───────────────────────────────────────────────────────
block(CX, 6.12, BW, BH, C_MLP,
      "MLP Head — Dense Layer 2",
      sub="Linear  32  →  1   |   mortality logit  (scalar, no activation)",
      fs=10)
arrow(CX, 5.79, 5.36, "(1,)  logit")

# ── 8. Sigmoid ───────────────────────────────────────────────────────────
block(CX, 5.03, 4.6, 0.56, C_INPUT,
      "Sigmoid  σ(logit)",
      fs=11)
arrow(CX, 4.75, 4.31, "∈ [0, 1]")

# ── 9. Output ────────────────────────────────────────────────────────────
block(CX, 3.96, BW, BH + 0.10, C_OUT,
      "Predicted Mortality Risk   p̂",
      sub="P(in-hospital death)  ∈  [0, 1]   —"
          "   threshold at Score1-optimal operating point",
      fs=10.5)

# ── Caption ──────────────────────────────────────────────────────────────
caption = (
    "Input (48×74) projected to d = 64 via a shared linear layer, summed with learnable sinusoidal TAPE,\n"
    "prepended with a [CLS] token, and passed through 4 pre-norm encoder layers (total 205 153 parameters).\n"
    "The contextualised CLS vector feeds a two-layer MLP (64 → 32 → 1) that emits the mortality logit."
)
ax.text(CX, 2.62, caption,
        ha="center", va="center", fontsize=9.5, style="italic",
        color="#333333", linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.55",
                  facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=1.0),
        zorder=3)

plt.tight_layout(pad=0.2)

# ── Save ─────────────────────────────────────────────────────────────────
png_path = OUT_DIR / "fig2_2_transformer_FIXED.png"
pdf_path = OUT_DIR / "fig2_2_transformer_FIXED.pdf"
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"[ok] {png_path}")
print(f"[ok] {pdf_path}")
