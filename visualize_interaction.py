import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.widgets import CheckButtons, Slider
import numpy as np


def visualize_satellites(sat_dfs, joint_exclusion=True):
    """
    Interactive GNSS visualizer with:
      - Skyplot (left)
      - Time series (right)
      - Satellite checkboxes
      - Measurement checkboxes
      - Consistent satellite colors
      - Time slider (shows current-time markers + vertical line)
      - Hover highlight (track + time-series + tooltip)
    sat_dfs: dict[label_or_id -> DataFrame] with columns:
      'gps_millis', 'Cn0DbHz', 'raw_pr_m' (optional), 'PseudorangeRateMetersPerSecond' (optional),
      'elevation_deg', 'azimuth_deg'
    """

    # ------------------------------------------------------------------
    # Normalize satellite keys to strings & build mapping
    # ------------------------------------------------------------------
    sat_keys = list(sat_dfs.keys())
    sat_labels = [str(k) for k in sat_keys]
    sat_key_map = {str(k): k for k in sat_keys}  # label -> original key

    # -----------------------------------------------------------
    # FILTER: keep only satellites that have at least one valid
    # elevation & azimuth entry
    # -----------------------------------------------------------
    valid_sat_labels = []
    for label in sat_labels:
        key = sat_key_map[label]
        df = sat_dfs[key]

        # If all elevation or all azimuth is NaN, skip it
        if df["elevation_deg"].isna().all() or df["azimuth_deg"].isna().all():
            print(f"[INFO] Skipping SVID {label}: no valid az/el data")
            continue

        valid_sat_labels.append(label)

    # If nothing valid, stop gracefully
    if not valid_sat_labels:
        print("No satellites with valid azimuth/elevation. Nothing to visualize.")
        return

    # Use only this list from here on
    sat_labels = valid_sat_labels

    # ALSO filter sat_keys to match valid_sat_labels
    sat_keys = [sat_key_map[label] for label in sat_labels]


    # ------------------------------------------------------------------
    # Figure & axes
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Colors per satellite
    # ------------------------------------------------------------------
    cmap = get_cmap("tab20")
    sat_colors = {label: cmap(i % 20) for i, label in enumerate(sat_labels)}

    # Storage for plot objects
    sky_lines = {}             # full track
    sat_markers = {}           # current-time markers (for slider)
    ts_lines = {               # measurement_type -> {sat_label -> Line2D}
        "CN0": {},
        "PR": {},
        "PRR": {},
        "EL": {},
        "AZ": {},
    }

    meas_labels = ["CN0", "PR", "PRR", "EL", "AZ"]

    # ------------------------------------------------------------------
    # Compute global time range & padding for slider
    # ------------------------------------------------------------------
    all_times = []
    for key in sat_keys:
        df = sat_dfs[key]
        if "gps_millis" in df.columns:
            all_times.append(df["gps_millis"].to_numpy())
    if not all_times:
        print("No gps_millis data available.")
        return

    all_times = np.concatenate(all_times)
    all_times = all_times[np.isfinite(all_times)]
    t_min, t_max = float(np.min(all_times)), float(np.max(all_times))
    pad_ms = 5000.0  # ±5 seconds
    slider_min = t_min - pad_ms
    slider_max = t_max + pad_ms
    initial_t = t_min

    # ------------------------------------------------------------------
    # Create plots for each satellite
    # ------------------------------------------------------------------
    for label in sat_labels:
        key = sat_key_map[label]
        df = sat_dfs[key].dropna(subset=["azimuth_deg", "elevation_deg"])
        if df.empty:
            continue

        color = sat_colors[label]

        theta = np.deg2rad(df["azimuth_deg"].to_numpy())
        r = 90.0 - df["elevation_deg"].to_numpy()
        x = df["gps_millis"].to_numpy()

        # Full track on skyplot
        sky_lines[label], = ax_sky.plot(theta, r, ".", color=color)

        # Marker for "current time" position (initially hidden)
        sat_markers[label], = ax_sky.plot([], [], marker="o", markersize=8,
                                          color=color, linestyle="None", alpha=0.0)

        # Time-series lines (start alpha=0; measurement toggles control visibility)
        ts_lines["CN0"][label], = ax_ts.plot(x, df["Cn0DbHz"].to_numpy(),
                                             color=color, alpha=0.0)
        ts_lines["PR"][label], = ax_ts.plot(
            x,
            df.get("raw_pr_m", np.full_like(x, np.nan)),
            color=color,
            alpha=0.0,
        )
        ts_lines["PRR"][label], = ax_ts.plot(
            x,
            df.get("PseudorangeRateMetersPerSecond", np.full_like(x, np.nan)),
            color=color,
            alpha=0.0,
        )
        ts_lines["EL"][label], = ax_ts.plot(x, df["elevation_deg"].to_numpy(),
                                            color=color, alpha=0.0)
        ts_lines["AZ"][label], = ax_ts.plot(x, df["azimuth_deg"].to_numpy(),
                                            color=color, alpha=0.0)

    # ------------------------------------------------------------------
    # Checkbox panels
    # ------------------------------------------------------------------
    # Satellite checkboxes
    rax_sat = plt.axes([0.01, 0.15, 0.12, 0.7])
    sat_check = CheckButtons(rax_sat, sat_labels, [True] * len(sat_labels))

    # Measurement checkboxes (CN0 on by default)
    rax_meas = plt.axes([0.88, 0.4, 0.1, 0.4])
    meas_check = CheckButtons(rax_meas, meas_labels, [True, False, False, False, False])

    # Time slider axis
    slider_ax = plt.axes([0.25, 0.05, 0.5, 0.03])
    time_slider = Slider(slider_ax, "Time (ms)", slider_min, slider_max, valinit=initial_t)

    # Vertical line in time-series at current time
    time_line = ax_ts.axvline(initial_t, color="k", linestyle="--", alpha=0.7)

    # Hover tooltip
    tooltip = ax_sky.text(
        0.02,
        0.98,
        "",
        transform=ax_sky.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="w", alpha=0.7),
        visible=False,
    )
    highlighted_sat = {"label": None}

    # ------------------------------------------------------------------
    # Helper: get states
    # ------------------------------------------------------------------
    def get_sat_status():
        return dict(zip(sat_labels, sat_check.get_status()))

    def get_meas_status():
        return dict(zip(meas_labels, meas_check.get_status()))

    # ------------------------------------------------------------------
    # Autoscale Y-axis for visible time-series
    # ------------------------------------------------------------------
    def rescale_yaxis():
        ymin, ymax = None, None
        for cat in ts_lines:
            for line in ts_lines[cat].values():
                if line.get_alpha() > 0.0:
                    y = line.get_ydata()
                    y = y[np.isfinite(y)]
                    if y.size == 0:
                        continue
                    lo, hi = float(np.min(y)), float(np.max(y))
                    if ymin is None:
                        ymin, ymax = lo, hi
                    else:
                        ymin = min(ymin, lo)
                        ymax = max(ymax, hi)

        if ymin is None:
            ax_ts.set_ylim(-1, 1)
        else:
            if ymin == ymax:
                ymax += 1.0
            margin = 0.05 * (ymax - ymin)
            ax_ts.set_ylim(ymin - margin, ymax + margin)

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Core: update timeseries visibility from checkboxes
    # ------------------------------------------------------------------
    def update_timeseries_visibility():
        sat_state = get_sat_status()
        meas_state = get_meas_status()

        for cat in meas_labels:
            cat_on = meas_state[cat]
            for label in sat_labels:
                line = ts_lines[cat][label]
                if not cat_on:
                    line.set_alpha(0.0)
                else:
                    if joint_exclusion:
                        line.set_alpha(1.0 if sat_state[label] else 0.0)
                    else:
                        line.set_alpha(1.0)

        rescale_yaxis()

    # ------------------------------------------------------------------
    # Time slider update: markers + vertical line
    # ------------------------------------------------------------------
    def update_time(val):
        t = time_slider.val
        time_line.set_xdata([t, t])

        sat_state = get_sat_status()

        time_window_ms = 2000.0  # how close a point must be to be "present" at this time

        for label in sat_labels:
            key = sat_key_map[label]
            df = sat_dfs[key]
            if "gps_millis" not in df.columns or df.empty:
                sat_markers[label].set_alpha(0.0)
                continue

            times = df["gps_millis"].to_numpy()
            if times.size == 0:
                sat_markers[label].set_alpha(0.0)
                continue

            idx = int(np.argmin(np.abs(times - t)))
            dt = abs(times[idx] - t)

            if dt <= time_window_ms and sat_state[label]:
                az = df["azimuth_deg"].iloc[idx]
                el = df["elevation_deg"].iloc[idx]
                theta = np.deg2rad(az)
                r = 90.0 - el

                sat_markers[label].set_data([theta], [r])
                sat_markers[label].set_alpha(1.0)
            else:
                sat_markers[label].set_alpha(0.0)

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks: satellite & measurement checkboxes
    # ------------------------------------------------------------------
    def toggle_sat(label):
        # label is already string
        state = get_sat_status()[label]
        sky_lines[label].set_visible(state)
        # if a satellite is turned off, also hide its marker
        if not state:
            sat_markers[label].set_alpha(0.0)

        if joint_exclusion:
            update_timeseries_visibility()
        else:
            rescale_yaxis()

        fig.canvas.draw_idle()

    def toggle_meas(cat):
        update_timeseries_visibility()

    # ------------------------------------------------------------------
    # Hover highlight on skyplot
    # ------------------------------------------------------------------
    def on_hover(event):
        if event.inaxes != ax_sky:
            # clear highlight when moving off skyplot
            if highlighted_sat["label"] is not None:
                reset_highlight()
            return

        # Find nearest marker under cursor (only consider visible markers)
        closest_label = None
        min_dist = np.inf

        for label in sat_labels:
            marker = sat_markers[label]
            if marker.get_alpha() <= 0.0:
                continue
            xdata, ydata = marker.get_xdata(), marker.get_ydata()
            if len(xdata) == 0:
                continue

            # Convert (theta,r) of marker to display coords
            theta = xdata[0]
            r = ydata[0]
            mx, my = ax_sky.transData.transform((theta, r))
            dx = event.x - mx
            dy = event.y - my
            dist = np.hypot(dx, dy)
            if dist < min_dist and dist < 15:  # 15 pixels threshold
                min_dist = dist
                closest_label = label

        if closest_label is None:
            if highlighted_sat["label"] is not None:
                reset_highlight()
            return

        if highlighted_sat["label"] != closest_label:
            apply_highlight(closest_label)

    def reset_highlight():
        label = highlighted_sat["label"]
        if label is None:
            return

        # reset line widths
        sky_lines[label].set_linewidth(1.0)
        for cat in meas_labels:
            ts_lines[cat][label].set_linewidth(1.0)

        tooltip.set_visible(False)
        highlighted_sat["label"] = None
        fig.canvas.draw_idle()

    def apply_highlight(label):
        # reset old
        reset_highlight()

        highlighted_sat["label"] = label

        # thicken sky track & time-series lines
        sky_lines[label].set_linewidth(3.0)
        for cat in meas_labels:
            line = ts_lines[cat][label]
            if line.get_alpha() > 0.0:
                line.set_linewidth(2.5)

        # build tooltip text from current slider time
        key = sat_key_map[label]
        df = sat_dfs[key]
        if df.empty or "gps_millis" not in df.columns:
            return

        t = time_slider.val
        times = df["gps_millis"].to_numpy()
        idx = int(np.argmin(np.abs(times - t)))
        az = df["azimuth_deg"].iloc[idx]
        el = df["elevation_deg"].iloc[idx]
        cn0 = df["Cn0DbHz"].iloc[idx]

        text = f"SV {label}\n" \
               f"t = {times[idx]:.0f} ms\n" \
               f"El = {el:.1f}°  Az = {az:.1f}°\n" \
               f"C/N0 = {cn0:.1f} dB-Hz"

        tooltip.set_text(text)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Wire everything up
    # ------------------------------------------------------------------
    time_slider.on_changed(update_time)
    sat_check.on_clicked(toggle_sat)
    meas_check.on_clicked(toggle_meas)
    cid_hover = fig.canvas.mpl_connect("motion_notify_event", on_hover)

    # Initial setup
    update_timeseries_visibility()
    update_time(initial_t)

    plt.show()
