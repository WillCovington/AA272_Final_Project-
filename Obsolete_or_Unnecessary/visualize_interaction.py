import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import CheckButtons
import numpy as np


def visualize_satellites(sat_dfs, joint_exclusion=True):
    """
    Clean, stable interactive GNSS visualizer with:
      • skyplot
      • time-series
      • consistent colors
      • sat toggles
      • measurement toggles
      • optional joint-exclusion
      • autoscaling
      • fixed checkbox behavior
    """

    # ------------------------------------------------------------
    # FIGURE + AXES
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))

    ax_sky = fig.add_subplot(121, projection="polar")
    ax_sky.set_title("GNSS Skyplot")
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(90, 0)
    ax_sky.grid(True)

    ax_ts = fig.add_subplot(122)
    ax_ts.set_title("Time Series")
    ax_ts.set_xlabel("GPS Millis")
    ax_ts.grid(True)

    # ------------------------------------------------------------
    # SATELLITE LABELS (STRING VERSION)
    # ------------------------------------------------------------
    sat_labels = [str(svid) for svid in sat_dfs.keys()]

    # assign colors
    cmap = get_cmap("tab20")
    sat_colors = {label: cmap(i % 20) for i, label in enumerate(sat_labels)}

    # dicts storing plot line handles
    sky_lines = {}
    ts_lines = {"CN0": {}, "PR": {}, "PRR": {}, "EL": {}, "AZ": {}}

    # ------------------------------------------------------------
    # CREATE ALL PLOTS
    # ------------------------------------------------------------
    for svid_str in sat_labels:
        df = sat_dfs[int(svid_str)]
        df = df.dropna(subset=["azimuth_deg", "elevation_deg"])
        if df.empty:
            continue

        color = sat_colors[svid_str]

        theta = np.deg2rad(df["azimuth_deg"])
        r = 90 - df["elevation_deg"]

        # Skyplot
        sky_lines[svid_str], = ax_sky.plot(theta, r, ".", color=color)

        # Timeseries (all invisible to begin)
        x = df["gps_millis"].to_numpy()

        ts_lines["CN0"][svid_str], = ax_ts.plot(x, df["Cn0DbHz"], color=color, alpha=0.0)
        ts_lines["PR"][svid_str],  = ax_ts.plot(x, df.get("raw_pr_m", np.nan), color=color, alpha=0.0)
        ts_lines["PRR"][svid_str], = ax_ts.plot(x, df.get("PseudorangeRateMetersPerSecond", np.nan), color=color, alpha=0.0)
        ts_lines["EL"][svid_str],  = ax_ts.plot(x, df["elevation_deg"], color=color, alpha=0.0)
        ts_lines["AZ"][svid_str],  = ax_ts.plot(x, df["azimuth_deg"], color=color, alpha=0.0)

    # ------------------------------------------------------------
    # CHECKBOXES
    # ------------------------------------------------------------
    # satellites
    rax_sat = plt.axes([0.01, 0.15, 0.12, 0.7])
    sat_check = CheckButtons(rax_sat, sat_labels, [True] * len(sat_labels))

    # measurement categories
    rax_meas = plt.axes([0.88, 0.4, 0.1, 0.4])
    meas_labels = ["CN0", "PR", "PRR", "EL", "AZ"]
    meas_check = CheckButtons(rax_meas, meas_labels, [False] * len(meas_labels))

    # ------------------------------------------------------------
    # Y-AUTOSCALING
    # ------------------------------------------------------------
    def rescale_yaxis():
        ymin, ymax = None, None

        for cat in ts_lines:
            for line in ts_lines[cat].values():
                if line.get_alpha() > 0:
                    y = line.get_ydata()
                    y = y[np.isfinite(y)]
                    if y.size == 0:
                        continue
                    lo, hi = np.min(y), np.max(y)
                    if ymin is None:
                        ymin, ymax = lo, hi
                    else:
                        ymin = min(ymin, lo)
                        ymax = max(ymax, hi)

        if ymin is None:
            ax_ts.set_ylim(-1, 1)
        else:
            if ymin == ymax:
                ymax += 1
            margin = 0.05 * (ymax - ymin)
            ax_ts.set_ylim(ymin - margin, ymax + margin)

        fig.canvas.draw_idle()

    # ------------------------------------------------------------
    # CALLBACK: SATELLITE TOGGLE
    # ------------------------------------------------------------
    def toggle_sat(label):
        # current state (True if checkbox is now ON)
        state = sat_check.get_status()[sat_labels.index(label)]

        # show/hide skyplot
        sky_lines[label].set_visible(state)

        # update time-series if joint mode
        if joint_exclusion:
            for cat in meas_labels:
                meas_on = meas_check.get_status()[meas_labels.index(cat)]
                line = ts_lines[cat][label]
                if meas_on and state:
                    line.set_alpha(1.0)
                else:
                    line.set_alpha(0.0)

        rescale_yaxis()

    # ------------------------------------------------------------
    # CALLBACK: MEASUREMENT TOGGLE
    # ------------------------------------------------------------
    def toggle_meas(cat):
        idx = meas_labels.index(cat)
        state = meas_check.get_status()[idx]    # True means enabled

        for svid_str in sat_labels:
            sat_visible = sky_lines[svid_str].get_visible()
            line = ts_lines[cat][svid_str]

            if state:
                # show if satellite also visible OR if joint mode disabled
                if sat_visible or not joint_exclusion:
                    line.set_alpha(1.0)
                else:
                    line.set_alpha(0.0)
            else:
                # hide always
                line.set_alpha(0.0)

        rescale_yaxis()

    # ------------------------------------------------------------
    # CONNECT EVENTS
    # ------------------------------------------------------------
    sat_check.on_clicked(toggle_sat)
    meas_check.on_clicked(toggle_meas)

    plt.show()
