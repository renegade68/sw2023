"""
viz_rotation_3d.py - Production-frontier version of the SW(2023)
direction-vector rotation figure for p=1, q=2.

The frontier is a radial two-output production surface:

    (Y1, Y2) = A X^alpha (cos theta, sin theta), theta in [0, pi/2].

Thus larger input X expands the feasible output radius, while Y1 and Y2
represent alternative output mixes. The same SW direction-vector rotation
maps the original (X, Y1, Y2) space to rotated (Z1, Z2, U) space.

Usage:
    MPLBACKEND=Agg python3 viz_rotation_3d.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import null_space


plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333",
    "axes.linewidth": 0.8,
})


np.random.seed(42)
n = 80
A = 0.92
alpha = 0.58


def frontier_radius(x):
    return A * np.power(x, alpha)


# ---------------------------------------------------------------------------
# DGP: draw frontier points from a one-input, two-output radial frontier.
# ---------------------------------------------------------------------------
x_star = np.random.uniform(0.12, 1.0, n)
theta = np.random.uniform(0.10, np.pi / 2 - 0.10, n)
radius = frontier_radius(x_star)
y1_star = radius * np.cos(theta)
y2_star = radius * np.sin(theta)

# Direction vector: negative input component, positive output components.
d_raw = np.array([-np.mean(x_star), np.mean(y1_star), np.mean(y2_star)])
d = d_raw / np.linalg.norm(d_raw)

# Inefficiency: move observations away from the frontier in the -d direction.
# This increases input use and reduces outputs. The scale is deliberately
# visible so the geometric projection can be read in the figure.
eta = np.abs(np.random.randn(n)) * 0.055 + 0.020
X = x_star - eta * d[0]
Y1 = y1_star - eta * d[1]
Y2 = y2_star - eta * d[2]

# ---------------------------------------------------------------------------
# Rotation matrix R = [v1; v2; d].
# ---------------------------------------------------------------------------
V_ns = null_space(d.reshape(1, -1))
v1 = V_ns[:, 0]
v2 = V_ns[:, 1]
if v1[0] < 0:
    v1 = -v1
if v2[1] < 0:
    v2 = -v2
R = np.vstack([v1, v2, d])

W_obs = np.column_stack([X, Y1, Y2])
W_front = np.column_stack([x_star, y1_star, y2_star])
WR_obs = np.einsum("ij,kj->ik", W_obs, R, optimize=True)
WR_front = np.einsum("ij,kj->ik", W_front, R, optimize=True)
Z1_rot, Z2_rot, U_rot = WR_obs[:, 0], WR_obs[:, 1], WR_obs[:, 2]
Z1_front, Z2_front, U_front = WR_front[:, 0], WR_front[:, 1], WR_front[:, 2]

# ---------------------------------------------------------------------------
# Frontier surface grid.
# ---------------------------------------------------------------------------
Ng = 42
x_g = np.linspace(0.08, 1.02, Ng)
theta_g = np.linspace(0, np.pi / 2, Ng)
X_s, TH = np.meshgrid(x_g, theta_g)
R_s = frontier_radius(X_s)
Y1_s = R_s * np.cos(TH)
Y2_s = R_s * np.sin(TH)

surf_pts = np.column_stack([X_s.ravel(), Y1_s.ravel(), Y2_s.ravel()])
surf_rot = np.einsum("ij,kj->ik", surf_pts, R, optimize=True)
Z1_s = surf_rot[:, 0].reshape(Ng, Ng)
Z2_s = surf_rot[:, 1].reshape(Ng, Ng)
U_s = surf_rot[:, 2].reshape(Ng, Ng)

hi_idx = [5, 22, 44, 68]
hi_col = ["#e74c3c", "#e67e22", "#27ae60", "#8e44ad"]


fig = plt.figure(figsize=(15, 7.2))
ax_a = fig.add_subplot(121, projection="3d", computed_zorder=False)
ax_b = fig.add_subplot(122, projection="3d", computed_zorder=False)

# ---------------------------------------------------------------------------
# Panel A: original production space.
# ---------------------------------------------------------------------------
ax_a.plot_surface(X_s, Y1_s, Y2_s, color="steelblue", alpha=0.15,
                  linewidth=0, antialiased=True, shade=False)
ax_a.plot_wireframe(X_s, Y1_s, Y2_s, color="#2471a3", alpha=0.24, lw=0.35)
ax_a.scatter(X, Y1, Y2, s=11, color="steelblue", alpha=0.35, depthshade=True)

cx = np.array([np.mean(X), np.mean(Y1), np.mean(Y2)])
sc_d = 0.34
tail_d = cx - sc_d * d
head_d = cx + sc_d * d
vec_d = head_d - tail_d
ax_a.quiver(tail_d[0], tail_d[1], tail_d[2],
            vec_d[0], vec_d[1], vec_d[2],
            color="#c0392b", lw=1.8, arrow_length_ratio=0.22,
            normalize=False, zorder=16)
ax_a.text(head_d[0] + 0.035, head_d[1] + 0.02, head_d[2] + 0.02,
          r"$\mathbf{d}$", color="#c0392b", fontsize=16,
          fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                    edgecolor="none", alpha=0.72))

sc_p = 0.20
corners = [
    cx + sc_p * v1 + sc_p * v2,
    cx - sc_p * v1 + sc_p * v2,
    cx - sc_p * v1 - sc_p * v2,
    cx + sc_p * v1 - sc_p * v2,
]
plane = Poly3DCollection([corners], alpha=0.14, facecolor="#f7dc6f",
                         edgecolor="#b7950b", linewidth=0.9)
ax_a.add_collection3d(plane)

for vi, col_v, lbl in [(v1, "#555555", r"$\mathbf{v}_1$"),
                       (v2, "#888888", r"$\mathbf{v}_2$")]:
    tip_v = cx + sc_p * vi
    ax_a.plot([cx[0], tip_v[0]], [cx[1], tip_v[1]], [cx[2], tip_v[2]],
              "--", color=col_v, lw=1.6)
    ax_a.text(tip_v[0] + 0.02, tip_v[1] + 0.01, tip_v[2] + 0.01,
              lbl, color=col_v, fontsize=12,
              bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                        edgecolor="none", alpha=0.62))

for idx, col in zip(hi_idx, hi_col):
    wi = W_obs[idx]
    ws = W_front[idx]
    ax_a.scatter(wi[0], wi[1], wi[2], s=55, color=col,
                 edgecolors="white", linewidths=0.8,
                 depthshade=False, zorder=10)
    ax_a.scatter(ws[0], ws[1], ws[2], s=38, color=col, marker="D",
                 alpha=0.85, depthshade=False, zorder=10)
    ax_a.plot([wi[0], ws[0]], [wi[1], ws[1]], [wi[2], ws[2]],
              "-", color=col, lw=1.4)

# Small in-panel key for the paired markers. Use 2D axes coordinates to avoid
# 3D clipping and accidental overlap with the direction vector.
ax_a.text2D(0.04, 0.82,
            "circle: observation\n"
            "diamond: frontier point\n"
            "colored segment: projection",
            transform=ax_a.transAxes,
            fontsize=9, color="#333", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.88))

larg_a = hi_idx[int(np.argmax([eta[i] for i in hi_idx]))]
mid_a = (W_obs[larg_a] + W_front[larg_a]) / 2
lbl_ax, lbl_ay1, lbl_ay2 = 0.96, 0.10, 0.08
ax_a.text(lbl_ax, lbl_ay1, lbl_ay2, r"projection along $\mathbf{d}$",
          fontsize=10, color="#555", ha="center",
          bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                    edgecolor="#aaa", alpha=0.88))
ax_a.plot([lbl_ax - 0.04, mid_a[0]], [lbl_ay1 + 0.04, mid_a[1]],
          [lbl_ay2 + 0.04, mid_a[2]], "-", color="#999", lw=0.9)

ax_a.set_xlabel(r"Input $X$", fontsize=13, labelpad=8)
ax_a.set_ylabel(r"Output $Y_1$", fontsize=13, labelpad=8)
ax_a.set_zlabel(r"Output $Y_2$", fontsize=13, labelpad=8)
ax_a.set_xlim(0.05, 1.08)
ax_a.set_ylim(0, 1.0)
ax_a.set_zlim(0, 1.0)
ax_a.tick_params(axis="both", labelsize=9)
ax_a.set_title("(A)  Original production space  $(X,\\,Y_1,\\,Y_2)$\n"
               r"Project observations to the frontier along $\mathbf{d}$",
               fontsize=12, pad=10)
ax_a.view_init(elev=20, azim=-58)
for pane in [ax_a.xaxis.pane, ax_a.yaxis.pane, ax_a.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#cccccc")
ax_a.grid(True, alpha=0.22)

# ---------------------------------------------------------------------------
# Panel B: rotated regression space.
# ---------------------------------------------------------------------------
ax_b.plot_surface(Z1_s, Z2_s, U_s, color="steelblue", alpha=0.17,
                  linewidth=0, antialiased=True, shade=False)
ax_b.plot_wireframe(Z1_s, Z2_s, U_s, color="#2471a3", alpha=0.25, lw=0.35)

z1_lo, z1_hi = Z1_rot.min() - 0.08, Z1_rot.max() + 0.08
z2_lo, z2_hi = Z2_rot.min() - 0.08, Z2_rot.max() + 0.08
u_lo, u_hi = U_rot.min() - 0.16, U_s.max() + 0.07
u_base = u_lo + 0.04
base_verts = [[(z1_lo, z2_lo, u_base), (z1_hi, z2_lo, u_base),
               (z1_hi, z2_hi, u_base), (z1_lo, z2_hi, u_base)]]
base_patch = Poly3DCollection(base_verts, alpha=0.10, facecolor="#d0e8f0",
                              edgecolor="#7fb3c8", linewidth=0.6, zorder=1)
ax_b.add_collection3d(base_patch)
ax_b.text((z1_lo + z1_hi) / 2, z2_hi - 0.04, u_base + 0.12,
          r"$(Z_1,\,Z_2)$ covariate plane", fontsize=9,
          color="#2471a3", ha="center",
          bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                    edgecolor="none", alpha=0.72))

ax_b.scatter(Z1_rot, Z2_rot, U_rot, s=11, color="steelblue",
             alpha=0.38, depthshade=True)

ax_b.text2D(0.04, 0.82,
            "same colored points after rotation\n"
            r"projection becomes a vertical $U$ gap",
            transform=ax_b.transAxes,
            fontsize=9, color="#333", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.88))
ax_b.text2D(0.61, 0.18,
            r"direction $\mathbf{d}$ becomes the $U$-axis",
            transform=ax_b.transAxes,
            fontsize=9, color="#7b241c", ha="left",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#d7b0a6", alpha=0.90))

largest_hi = hi_idx[int(np.argmax([eta[i] for i in hi_idx]))]
for idx, col in zip(hi_idx, hi_col):
    z1i, z2i, ui = Z1_rot[idx], Z2_rot[idx], U_rot[idx]
    u_fr = U_front[idx]
    ax_b.scatter(z1i, z2i, u_base, s=28, color=col, marker="o",
                 alpha=0.55, depthshade=False, zorder=8)
    ax_b.plot([z1i, z1i], [z2i, z2i], [u_base, ui],
              ":", color=col, lw=1.1, alpha=0.55)
    ax_b.scatter(z1i, z2i, ui, s=55, color=col, edgecolors="white",
                 linewidths=0.8, depthshade=False, zorder=10)
    ax_b.plot([z1i, z1i], [z2i, z2i], [ui, u_fr],
              "--", color=col, lw=1.8, zorder=9)
    ax_b.scatter(z1i, z2i, u_fr, s=38, color=col, marker="D",
                 alpha=0.85, depthshade=False, zorder=10)

z1l, z2l = Z1_rot[largest_hi], Z2_rot[largest_hi]
gap_mid = (U_rot[largest_hi] + U_front[largest_hi]) / 2
lbl_z1 = z1_hi - 0.09
lbl_z2 = z2_lo + 0.05
lbl_u = u_lo + 0.16
ax_b.text(lbl_z1, lbl_z2, lbl_u, r"$\eta_i - \varepsilon_i$",
          fontsize=12, color="#222", ha="center",
          bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffde7",
                    edgecolor="#aaa", alpha=0.95))
ax_b.plot([lbl_z1 - 0.04, z1l], [lbl_z2 + 0.06, z2l],
          [lbl_u + 0.08, gap_mid], "-", color="#888", lw=1.0)

ax_b.set_xlim(z1_lo, z1_hi)
ax_b.set_ylim(z2_lo, z2_hi)
ax_b.set_zlim(u_lo, u_hi)
ax_b.set_xlabel(r"$Z_1$", fontsize=14, labelpad=8)
ax_b.set_ylabel(r"$Z_2$", fontsize=14, labelpad=8)
ax_b.set_zlabel(r"$U$", fontsize=14, labelpad=8)
ax_b.tick_params(axis="both", labelsize=9)
ax_b.set_title("(B)  Rotated space  $(Z_1,\\,Z_2,\\,U)$\n"
               r"Frontier becomes the regression surface $\varphi(Z_1,Z_2)$",
               fontsize=12, pad=10)
ax_b.view_init(elev=28, azim=30)
for pane in [ax_b.xaxis.pane, ax_b.yaxis.pane, ax_b.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#cccccc")
ax_b.grid(True, alpha=0.22)

fig.text(0.500, 0.89, r"$R_{\mathbf{d}}\ \longrightarrow$",
         ha="center", va="top", fontsize=15,
         color="#1a5276", fontweight="bold")
fig.text(0.500, 0.845, "rotate the frontier surface and data",
         ha="center", va="top", fontsize=9, color="#555555")

fig.savefig("fig_rotation_3d.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("fig_rotation_3d.png", bbox_inches="tight", dpi=180,
            facecolor="white")
print("Saved: fig_rotation_3d.pdf, fig_rotation_3d.png")
