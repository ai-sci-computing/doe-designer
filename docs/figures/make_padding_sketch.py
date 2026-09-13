"""Draws docs/figures/padding.png: side view of the computational box (the padded window along
x, the distance d along z, not to scale). The plate occupies the central N p of the entry face,
the picture the central N p of the exit face, and the light of the plate spreads by
d tan(theta_max) to each side on its way. The transform identifies the top and bottom edges of
the window (a torus): light that reaches one re-enters from the other. The window is chosen so
that this wrapped light lands in the far don't-care strip and misses the picture,
S = N p + d tan(theta_max) (the picture-clear rule). Sizes of the default setup: N p = 4.1 mm,
d tan(theta_max) = 1.66 mm, S = 5.76 mm (720 px). Development-only (matplotlib); the PNG is
committed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

A = 4.096   # plate / picture, mm
R = 1.663   # reach of the light beyond the plate, mm
S = A + R   # padded window, mm: the picture-clear rule S = N p + d tan(theta_max)
D = 9.0     # drawing length of the distance d (not to scale)
h, s2 = A / 2, S / 2
blue, gray = "#3060c0", "#606060"

fig, ax = plt.subplots(figsize=(11, 6.4), dpi=200)
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-3.4, D + 3.0); ax.set_ylim(-s2 - 1.6, s2 + 1.6)

# the box: padded window (vertical) times the distance (horizontal)
ax.add_patch(Rectangle((0, -s2), D, S, fc="#f6f6f6", ec="k", lw=1.3))
# the light of the plate: from the plate's edges at z = 0 to +-(A/2 + R) at z = d
# inside the window the cone is clipped at the edges; beyond them it re-enters from the opposite edge
cone = Polygon([[0, -h], [0, h], [D, h + R], [D, -h - R]], closed=True, fc=blue, ec="none", alpha=0.25)
ax.add_patch(cone); cone.set_clip_path(Rectangle((0, -s2), D, S, transform=ax.transData))
zc = D * (s2 - h) / R   # where the cone edge crosses the window edge
for sign in (1, -1):
    ax.add_patch(Polygon([[zc, -sign * s2], [D, -sign * s2], [D, -sign * (s2 - R / 2)]], closed=True, fc=blue, ec="none", alpha=0.25, hatch="///"))
# the box the viewer shows: a crop of 384 px (default at N = 512) around the axis over the whole distance
V = A * 384 / 512   # the native viewer's default crop, 384 px of the 512 px plate
ax.add_patch(Rectangle((0, -V / 2), D, V, fc="none", ec="#2a4fa0", lw=1.2, ls="--"))
ax.text(D / 2, V / 2 - 0.25, "box shown by the viewer (384 px around the axis by default)", ha="center", va="top", fontsize=8.5, color="#2a4fa0")
# plate on the entry face, picture on the exit face
ax.add_patch(Rectangle((-0.16, -h), 0.16, A, fc="#ffe0a0", ec="k", lw=1))
ax.add_patch(Rectangle((D, -h), 0.16, A, fc="#40c040", ec="k", lw=1))
ax.text(-0.35, 0, "plate\n$N p$ = 4.1 mm", ha="right", va="center", fontsize=9)
ax.text(D + 0.35, 0, "picture\n$N p$ = 4.1 mm", ha="left", va="center", fontsize=9)
ax.text(-0.35, h + (s2 - h) / 2, "padding,\nno plate", ha="right", va="center", fontsize=8, color=gray)
ax.text(-0.35, -h - (s2 - h) / 2, "padding,\nno plate", ha="right", va="center", fontsize=8, color=gray)
ax.text(D + 0.35, h + (s2 - h) / 2, "don't care", ha="left", va="center", fontsize=8, color=gray)
ax.text(D + 0.35, -h - (s2 - h) / 2, "don't care", ha="left", va="center", fontsize=8, color=gray)
ax.text(D / 2, 0, "light of the plate", ha="center", va="center", fontsize=9, color=blue)
ax.text(D - 0.2, -s2 + R / 4 + 0.05, "wrapped light,\nmisses the picture", ha="right", va="center", fontsize=8, color=blue)
# reach at the exit face: d tan(theta_max) beyond the plate edge, half of it beyond the window edge
ax.annotate("", xy=(D + 1.6, h), xytext=(D + 1.6, h + R), arrowprops=dict(arrowstyle="<->", color=blue, lw=1.1))
ax.text(D + 1.75, h + R / 2, r"$d\,\tan\theta_{\max}$ = 1.7 mm", ha="left", va="center", fontsize=8.5, color=blue)
ax.plot([D, D + 1.7], [h + R, h + R], color=blue, lw=0.8, ls=":")
# identified edges of the torus
for y, dy in ((s2, 1), (-s2, -1)):
    ax.add_patch(Polygon([[D / 2 - 0.25, y - 0.12], [D / 2 - 0.25, y + 0.12], [D / 2 + 0.1, y]], closed=True, fc="k", ec="none"))
ax.text(D / 2, s2 + 0.25, "top and bottom edge are identified: light that reaches one\nre-enters from the other", ha="center", va="bottom", fontsize=9)
# window size and distance
ax.annotate("", xy=(-2.6, -s2), xytext=(-2.6, s2), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
ax.text(-2.75, 0, r"window $S = N p + d\,\tan\theta_{\max}$ = 5.8 mm", ha="right", va="center", fontsize=9, rotation=90)
ax.annotate("", xy=(0, -s2 - 0.6), xytext=(D, -s2 - 0.6), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
ax.text(D / 2, -s2 - 0.75, "distance $d$ = 50 mm (not to scale)", ha="center", va="top", fontsize=9)
fig.savefig("docs/figures/padding.png", bbox_inches="tight", facecolor="white")
