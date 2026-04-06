########################## OUTDOOR OSL – GLOBAL MAP ##########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
plt.style.use('default')

# ─── STYLE ───────────────────────────────────────────────────────────────────

FONT_TS        = 30
TICK_TS        = 25
WIND_CMAP      = 'plasma'

FT_TO_M = 0.3048
FIELD_W = 40 * FT_TO_M
FIELD_H = 40 * FT_TO_M

# ─── FILES ───────────────────────────────────────────────────────────────────
files = {
    '38F, Avg Windspeed 3.1 m/s': (
        r'C:\Users\ltjth\Documents\Research\UKF_Data\ARR_outdoor_38F_alpha1_worked_passed_source.csv',
        5600
    ),
    '44F, Avg Windspeed 0.7 m/s': (
        r'C:\Users\ltjth\Documents\Research\UKF_Data\ARR_outdoor_44F4_initial approach worked_passed_source.csv',
        4400
    ),
}

# ─── PLACEMENT ───────────────────────────────────────────────────────────────
RUN_PLACEMENT = {
    '38F, Avg Windspeed 3.1 m/s': {'x_global': 2,   'y_global': 11,  'heading_deg': 315},
    '44F, Avg Windspeed 0.7 m/s': {'x_global': 1.5,   'y_global': 5, 'heading_deg': 0},
}

def rotate_xy(x, y, deg):
    rad = np.deg2rad(deg)
    return x * np.cos(rad) - y * np.sin(rad), x * np.sin(rad) + y * np.cos(rad)

def make_line_collection(x, y, values, norm, cmap, linewidth=2.5, zorder=3):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_vals = (values[:-1] + values[1:]) / 2
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, zorder=zorder)
    lc.set_array(seg_vals)
    return lc

# ─── FIGURE ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 22), facecolor='white')
fig.subplots_adjust(wspace=0.35)

cmap = cm.get_cmap(WIND_CMAP)

for idx, (label, (path, split_idx)) in enumerate(files.items()):
    ax = axes[idx]
    ax.set_facecolor('white')
    placement = RUN_PLACEMENT[label]
    gx, gy, hdg = placement['x_global'], placement['y_global'], placement['heading_deg']

    df = pd.read_csv(path)
    x_loc      = df['PosX'].values - df['PosX'].values[0]
    y_loc      = df['PosY'].values - df['PosY'].values[0]
    flow_angle = df['FlowAngle'].values
    wind       = df['Gas'].values                          # raw wind for quiver onset check
    gas        = df['Gas'].values                          # gas drives colormap
    wind_smooth = pd.Series(wind).rolling(window=50, center=True, min_periods=1).mean().values
    n          = len(x_loc)
    split_idx  = min(split_idx, n - 1)

    x_rot, y_rot = rotate_xy(x_loc, y_loc, hdg)
    x_glob = x_rot + gx
    y_glob = y_rot + gy

    # ── Per-file norm based on Gas ────────────────────────────────────────────
    norm = mcolors.Normalize(vmin=gas.min(), vmax=gas.max())

    # ── First index where wind > 0.03 ────────────────────────────────────────
    wind_onset_indices = np.where(wind > 0.03)[0]
    quiver_start_idx   = wind_onset_indices[0] if len(wind_onset_indices) > 0 else 0

    rect = plt.Rectangle((FIELD_W/2-0.35, FIELD_H/2-0.35), 0.5, 0.5,
                      linewidth=1.5, edgecolor='black',
                      facecolor='black', alpha=0.1, zorder=10)
    ax.add_patch(rect)

    # ── Dashed circle around source
    circle = mpatches.Circle(
        (FIELD_W/2-0.1, FIELD_H/2-0.1), radius=1,
        linewidth=1.8, edgecolor='black', facecolor='none',
        linestyle='--', zorder=5
    )
    ax.add_patch(circle)

    # ── Pre-source: gas-colored trajectory
    lc_pre = make_line_collection(
        x_glob[:split_idx+1], y_glob[:split_idx+1],
        gas[:split_idx+1], norm, cmap,
        linewidth=2.5, zorder=3
    )
    ax.add_collection(lc_pre)

    # ── Post-source: gray dashed
    ax.plot(x_glob[split_idx:], y_glob[split_idx:],
            color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=3)

    # ── Markers
    ax.scatter(x_glob[0],         y_glob[0],
               color='white', marker='o', s=80, zorder=5, edgecolors='black', linewidths=1.2)
    ax.scatter(x_glob[split_idx], y_glob[split_idx],
               color='gold',  marker='*', s=280, zorder=6, edgecolors='black', linewidths=0.8)
    ax.scatter(x_glob[-1],        y_glob[-1],
               color='white', marker='s', s=100, zorder=6, edgecolors='black', linewidths=1.2)

    # ── Wind direction arrows (only after quiver_start_idx, thicker)
    arrow_len = 0.3
    for i in range(0, len(x_glob), 200):
        if i < quiver_start_idx:
            continue
        angle_rad = np.deg2rad(flow_angle[i] + hdg + 180)
        dx = np.cos(angle_rad) * arrow_len
        dy = np.sin(angle_rad) * arrow_len
        ax.annotate('',
            xy=(x_glob[i] + dx, y_glob[i] + dy),
            xytext=(x_glob[i], y_glob[i]),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.8, mutation_scale=14),
            zorder=4)

    # ── Per-subplot colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04, fraction=0.03, aspect=30)
    cbar.set_label('Gas (a.u.)', fontsize=FONT_TS - 4)
    cbar.ax.tick_params(labelsize=TICK_TS - 4)

    # ── Legend
    legend_handles = [
        mlines.Line2D([], [], color='black', marker='o', markersize=7,
                      linewidth=0, markerfacecolor='white', markeredgecolor='black', label='Start'),
        mlines.Line2D([], [], color='black', marker='*', markersize=12,
                      linewidth=0, markerfacecolor='gold',  markeredgecolor='black', label='Passed Source'),
        mlines.Line2D([], [], color='black', marker='s', markersize=7,
                      linewidth=0, markerfacecolor='white', markeredgecolor='black', label='End'),
        mlines.Line2D([], [], color='gray',  linestyle='--', linewidth=1.5, label='After Source'),
        mlines.Line2D([], [], color='black', marker='>', markersize=6,
                      linewidth=1.0, label='Wind Dir.'),
    ]
    ax.legend(handles=legend_handles, fontsize=12, framealpha=0.9, loc='upper right')

    # ── Axes formatting
    ax.set_xlim(-0.5, FIELD_W + 0.5)
    ax.set_ylim(-0.5, FIELD_H + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)', fontsize=FONT_TS)
    ax.set_ylabel('Y (m)', fontsize=FONT_TS)
    ax.set_title(label, fontsize=FONT_TS)
    ax.tick_params(labelsize=TICK_TS)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray', zorder=1)

#plt.suptitle('Outdoor OSL — Global Map', fontsize=FONT_TS + 2, y=1.01)
plt.tight_layout()
plt.show()
