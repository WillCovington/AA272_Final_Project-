# ==========================================================
# plot_tools.py
# All LOS/NLOS time-series and 3D surface plotting
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from los_nlos_analysis import (
    simulate_constellations,
    building_shadow_angle,
    global_mean_event_gap,
)


# ==========================================================
# 2D LOS/NLOS plot
# ==========================================================
def plot_los_nlos_timeseries(sat_positions, times, L, X, AZ_b=0):
    theta_block = building_shadow_angle(L, X)
    sats = sorted(sat_positions.keys())

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, sid in enumerate(sats):
        az = np.mod(sat_positions[sid]["az"], 360)
        el = sat_positions[sid]["el"]

        los = (el > 0) & ((az < AZ_b) | (az > AZ_b + theta_block))
        nlos = (el > 0) & (~los)

        los_seg  = np.where(los,  i, np.nan)
        nlos_seg = np.where(nlos, i, np.nan)

        ax.plot(times, los_seg,  color="green", linewidth=3)
        ax.plot(times, nlos_seg, color="red",   linewidth=3)

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Satellite ID")
    ax.set_yticks(range(len(sats)))
    ax.set_yticklabels([f"Sat {s}" for s in sats])

    ax.grid(True)
    ax.set_title(f"LOS (green) vs NLOS (red)\nShadow width = {theta_block:.1f}°")
    plt.tight_layout()
    plt.show()


# ==========================================================
# 3D Surface Plot
# ==========================================================
def surface_plot_gap_times(
        sat_positions,
        times,
        L_values,
        X_values,
        H,
        AZ_b=0):

    L_grid, X_grid = np.meshgrid(L_values, X_values)
    T_grid = np.zeros_like(L_grid)

    for i in range(len(X_values)):
        for j in range(len(L_values)):
            L = L_grid[i, j]
            X = X_grid[i, j]
            T = global_mean_event_gap(sat_positions, times, L, X, H, AZ_b)
            T_grid[i, j] = T if not np.isnan(T) else np.nan

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(L_grid, X_grid, T_grid,
                           cmap=cm.viridis,
                           edgecolor="none")

    fig.colorbar(surf, label="Mean event gap (sec)")
    ax.set_xlabel("Building Width L (m)")
    ax.set_ylabel("Distance X (m)")
    ax.set_zlabel("Mean LOS↔NLOS event gap (sec)")
    ax.set_title("Impact of Building Geometry on Satellite Occlusion")

    plt.tight_layout()
    plt.show()


# ==========================================================
# Example driver
# ==========================================================
if __name__ == "__main__":
    # coordinates for stanford tanneer fountain
    lat = 37.4275
    lon = -122.1697

    sat_positions, times = simulate_constellations(lat, lon, duration_hours=12)

    # Example 2D plot
    plot_los_nlos_timeseries(sat_positions, times, L=18.1, X=50)

    # Example 3D plot
    L_vals = np.linspace(5, 80, 100)
    X_vals = np.linspace(10, 200, 100)
    surface_plot_gap_times(sat_positions, times, L_vals, X_vals, H=87)
