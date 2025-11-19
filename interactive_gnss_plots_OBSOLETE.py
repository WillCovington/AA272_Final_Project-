import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np


def visualize_satellites(sat_dfs, joint_exclusion=True):
    """
    Interactive skyplot + time-series with satellite toggles, measurement toggles,
    autoscaling, grid, and joint exclusion control.
    """

    fig = plt.figure(figsize=(14, 8))

    # ------------------------------------------------------------
    # SKY PLOT
    # ------------------------------------------------------------
    ax_sky = fig.add_subplot(121, projection="polar")
    ax_sky.set_title("GNSS Skyplot", fontsize=15)
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(90, 0)
    ax_sky.grid(True)

    # ------------------------------------------------------------
    # TIME SERIES PLOT
    # ------------------------------------------------------------
    ax_ts = fig.add_subplot(122)
    ax_ts.set_title("Time Series")
    ax_ts.set_xlabel("GPS Millis")
    ax_ts.grid(True)  # <--- GRID ADDED HERE

    sky_lines = {}
    ts_lines = {"CN0": {}, "PR": {}, "PRR": {}, "EL": {}, "AZ": {}}

    # ------------------------------------------------------------
    # CREATE PLOTS FOR EACH SATELLITE
    # ------------------------------------------------------------
    for svid, df in sat_dfs.items():
        df = df.dropna(subset=["azimuth_deg", "elevation_deg"])
        if df.empty:
            continue

        theta = np.deg2rad(df["azimuth_deg"])
        r = 90 - df["elevation_deg"]

        # skyplot
        sky_lines[str(svid)], = ax_sky.plot(theta, r, ".", label=str(svid))

        # time-series (initially hidden)
        ts_lines["CN0"][svid], = ax_ts.plot(df["gps_millis"], df["Cn0DbHz"],
                                            label=f"{svid} CN0", alpha=0.0)
        ts_lines["PR"][svid],  = ax_ts.plot(df["gps_millis"],
                                            df.get("raw_pr_m", np.nan),
                                            label=f"{svid} PR", alpha=0.0)
        ts_lines["PRR"][svid], = ax_ts.plot(df["gps_millis"],
                                            df.get("PseudorangeRateMetersPerSecond", np.nan),
                                            label=f"{svid} PRR", alpha=0.0)
        ts_lines["EL"][svid],  = ax_ts.plot(df["gps_millis"], df["elevation_deg"],
                                            label=f"{svid} Elev", alpha=0.0)
        ts_lines["AZ"][svid],  = ax_ts.plot(df["gps_millis"], df["azimuth_deg"],
                                            label=f"{svid} Azim", alpha=0.0)

    # ------------------------------------------------------------
    # CHECKBOX PANELS
    # ------------------------------------------------------------
    rax_sat = plt.axes([0.01, 0.15, 0.12, 0.7])
    sat_labels = list(sky_lines.keys())
    sat_vis = [True] * len(sat_labels)
    sat_check = CheckButtons(rax_sat, sat_labels, sat_vis)

    rax_meas = plt.axes([0.88, 0.4, 0.1, 0.4])
    meas_labels = ["CN0", "PR", "PRR", "EL", "AZ"]
    meas_vis = [False] * len(meas_labels)
    meas_check = CheckButtons(rax_meas, meas_labels, meas_vis)

    # ------------------------------------------------------------
    # AUTO-SCALING
    # ------------------------------------------------------------
    def rescale_timeseries_yaxis():
        ymin, ymax = None, None

        for cat in ts_lines:
            for svid, line in ts_lines[cat].items():
                if line.get_alpha() > 0.0:
                    y = line.get_ydata()
                    y = y[np.isfinite(y)]
                    if y.size == 0:
                        continue

                    if ymin is None:
                        ymin, ymax = np.min(y), np.max(y)
                    else:
                        ymin = min(ymin, np.min(y))
                        ymax = max(ymax, np.max(y))

        if ymin is None:
            ax_ts.set_ylim(0, 1)
        else:
            if ymin == ymax:
                ymax += 1
            margin = 0.05 * (ymax - ymin)
            ax_ts.set_ylim(ymin - margin, ymax + margin)

        ax_ts.figure.canvas.draw_idle()

    # ------------------------------------------------------------
    # CALLBACKS 
    # ------------------------------------------------------------

    def toggle_satellite(label):
        # Toggle satellite only
        sky_line = sky_lines[label]
        new_vis = not sky_line.get_visible()
        sky_line.set_visible(new_vis)

        if joint_exclusion:
            # Only modify lines for ENABLED measurement categories
            for cat in meas_labels:
                # category enabled?
                if meas_check.get_status()[meas_labels.index(cat)]:
                    if label in ts_lines[cat]:
                        ts_lines[cat][label].set_alpha(1.0 if new_vis else 0.0)

            rescale_timeseries_yaxis()

        plt.draw()

    def toggle_measurement(label):
        cat = label
        enabled = not meas_check.get_status()[meas_labels.index(cat)]

        # Update EVERY satellite in this category
        for svid, line in ts_lines[cat].items():
            sat_visible = sky_lines[str(svid)].get_visible()

            # If measurement category enabled:
            #   show satellite only if satellite is also enabled
            if enabled:
                if sat_visible or (not joint_exclusion):
                    line.set_alpha(1.0)
                else:
                    line.set_alpha(0.0)
            else:
                # Measurement category disabled: hide always
                line.set_alpha(0.0)

        rescale_timeseries_yaxis()
        plt.draw()

