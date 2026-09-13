"""Draws docs/figures/setup.png, the optical setup sketch of the manual.
Development-only (matplotlib); the PNG is committed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
import numpy as np

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=200)
ax.set_xlim(-0.5, 11.5); ax.set_ylim(-2.3, 1.9); ax.axis("off")

# laser + beam expander
ax.add_patch(Rectangle((-0.3, -0.25), 1.1, 0.5, fc="#c8c8c8", ec="k"))
ax.text(0.25, 0.5, "laser", ha="center", fontsize=10)
ax.add_patch(Polygon([[1.0, -0.15], [1.6, -0.55], [1.6, 0.55], [1.0, 0.15]], closed=True, fc="#e8f0ff", ec="k"))
ax.text(1.3, -1.2, "beam\nexpander", ha="center", va="top", fontsize=9)
# collimated beam (plane wave), wavefronts as vertical lines, as wide as the plate
for x in np.linspace(1.9, 3.3, 5):
    ax.plot([x, x], [-0.5, 0.5], color="#3060c0", lw=1)
ax.annotate("", xy=(3.55, 0), xytext=(1.8, 0), arrowprops=dict(arrowstyle="-|>", color="#3060c0", lw=1.5))
ax.text(2.6, 0.65, "plane wave\n(collimated, coherent,\nmonochromatic)", ha="center", va="bottom", fontsize=8.5)

# DOE: thin plate at z = 0 in the central half of the window's entry face, phase profile;
# the rest of the entry face is the zero padding of the computation (no plate, no light)
ax.add_patch(Rectangle((3.7, -1.0), 0.18, 2.0, fc="#e8e8e8", ec="#9a9a9a", ls=":"))
ax.add_patch(Rectangle((3.7, -0.5), 0.18, 1.0, fc="#ffe0a0", ec="k"))
zz = np.linspace(-0.47, 0.47, 40)
ax.plot(3.88 + 0.08 * (1 + np.cos(18 * zz)), zz, color="#a06000", lw=1)
ax.text(3.8, -1.2, "DOE (thin phase\nplate), $z = 0$", ha="center", va="top", fontsize=9)
ax.text(3.62, -0.75, "padding\n(zeros)", ha="right", va="center", fontsize=7.5, color="#707070")
ax.text(4.9, 1.05, r"$u = \mathrm{illum}(x,y)\,e^{i\varphi(x,y)}$", ha="left", va="bottom", fontsize=10)

# free-space box between DOE and target
box_x0, box_x1 = 3.95, 9.2
ax.add_patch(Rectangle((box_x0, -1.0), box_x1 - box_x0, 2.0, fc="#f4f8ff", ec="#8090b0", ls="--"))
ax.text((box_x0 + box_x1) / 2, -0.75, "free space, no lens: angular-spectrum propagation\n(the box the viewer shows)", ha="center", va="center", fontsize=9, color="#405070")
# a few diffracted rays
for y0, y1 in [(0.45, 0.2), (0.3, -0.5), (-0.2, 0.6), (-0.45, -0.1), (0.0, 0.05)]:
    ax.plot([box_x0, box_x1], [y0, y1], color="#3060c0", lw=0.8, alpha=0.7)
ax.annotate("", xy=(box_x1 - 0.1, -1.6), xytext=(box_x0 + 0.1, -1.6), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
ax.text((box_x0 + box_x1) / 2, -1.75, "distance $d$", ha="center", va="top", fontsize=9)

# target plane: screen / sensor
ax.add_patch(Rectangle((9.2, -1.0), 0.12, 2.0, fc="#d0d0d0", ec="k"))
ax.add_patch(Rectangle((9.2, -0.5), 0.12, 1.0, fc="#40c040", ec="k"))
ax.text(9.26, -1.2, "target plane (screen\nor sensor), $z = d$", ha="center", va="top", fontsize=9)
ax.text(10.4, 0.0, "central active x active\npixels: the image\n\nsurround: 'don't care'\nregion (stray light)", ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(9.35, 0.3), xytext=(9.75, 0.3), arrowprops=dict(arrowstyle="-|>", color="#208020", lw=1))
ax.annotate("", xy=(9.35, 0.85), xytext=(9.75, 0.85), arrowprops=dict(arrowstyle="-|>", color="#606060", lw=1))
ax.text(8.2, 1.05, r"$|v|^2 = |A u|^2$", ha="left", va="bottom", fontsize=10)
ax.annotate("", xy=(11.0, -2.15), xytext=(-0.3, -2.15), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))
ax.text(11.1, -2.15, "$z$", va="center", fontsize=10)
fig.savefig("docs/figures/setup.png", bbox_inches="tight")
print("wrote docs/figures/setup.png")
