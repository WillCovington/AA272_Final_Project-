import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm

# ==========================================================
# Earth constants (WGS-84)
# ==========================================================
MU_EARTH = 3.986004418e14        # m^3/s^2
R_EARTH = 6378137.0              # m
OMEGA_EARTH = 7.2921150e-5       # rad/s


# ==========================================================
# Rotation matrices
# ==========================================================
def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


# ==========================================================
# Kepler’s equation for circular orbits (e = 0)
# ==========================================================
def true_anomaly(M):
    return M   # simply equal for circular orbits


# ==========================================================
# Convert orbital → ECI
# CORRECT rotation order:
# r_eci = Rz(Ω) * Rx(i) * Rz(ω + ν) * [a, 0, 0]
# ==========================================================
def orbital_to_eci(a, inc, raan, argp, nu, e):
    # Compute radius for elliptical orbit
    r = a * (1 - e*e) / (1 + e*np.cos(nu))
    r_orb = np.array([r, 0, 0])
    u = argp + nu
    return Rz(raan) @ (Rx(inc) @ (Rz(u) @ r_orb))


# ==========================================================
# Convert ECI → ECEF
# ==========================================================
def eci_to_ecef(r_eci, t):
    return Rz(-OMEGA_EARTH * t) @ r_eci


# ==========================================================
# Convert ECEF → ENU → az/el
# ==========================================================
def ecef_to_az_el(r_ecef, lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # ECEF → ENU rotation
    R = np.array([
        [-np.sin(lon),              np.cos(lon),              0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])

    e, n, u = R @ r_ecef

    az = np.arctan2(e, n) % (2*np.pi)
    el = np.arctan2(u, np.sqrt(e*e + n*n))
    return np.degrees(az), np.degrees(el)


# ==========================================================
# Nominal constellations
# ==========================================================
import numpy as np

# -----------------------------------------------------------
# Slightly realistic GPS constellation
# -----------------------------------------------------------
def gps_constellation():
    sats = []
    a = R_EARTH + 20200e3
    inc = np.radians(55)
    planes = 6
    slots = 4
    raan_spacing = 360 / planes
    M_spacing = 360 / slots

    for p in range(planes):
        for s in range(slots):

            # Slight realism
            e = np.random.uniform(0.005, 0.02)                       # real GPS range
            raan = np.radians(p * raan_spacing) + np.radians(np.random.uniform(-3, 3))
            argp = np.radians(np.random.uniform(0, 360))
            M0 = np.radians(s * M_spacing) + np.radians(np.random.uniform(-5, 5))

            sats.append({
                "const": "GPS",
                "a": a,
                "inc": inc,
                "e": e,
                "raan": raan,
                "argp": argp,
                "M0": M0,
                "color": "tab:blue"
            })
    return sats


# -----------------------------------------------------------
# Slightly realistic Galileo constellation
# -----------------------------------------------------------
def galileo_constellation():
    sats = []
    a = R_EARTH + 23222e3
    inc = np.radians(56)
    planes = 3
    slots = 8
    raan_spacing = 360 / planes
    M_spacing = 360 / slots

    for p in range(planes):
        for s in range(slots):

            e = np.random.uniform(0.0005, 0.005)
            raan = np.radians(p * raan_spacing) + np.radians(np.random.uniform(-5, 5))
            argp = np.radians(np.random.uniform(0, 360))
            M0 = np.radians(s * M_spacing) + np.radians(np.random.uniform(-10, 10))

            sats.append({
                "const": "GAL",
                "a": a,
                "inc": inc,
                "e": e,
                "raan": raan,
                "argp": argp,
                "M0": M0,
                "color": "tab:green"
            })
    return sats


# -----------------------------------------------------------
# Slightly realistic BeiDou MEO constellation
# -----------------------------------------------------------
def beidou_meo_constellation():
    sats = []
    a = R_EARTH + 27878e3
    inc = np.radians(55)
    planes = 3
    slots = 8
    raan_spacing = 360 / planes
    M_spacing = 360 / slots

    for p in range(planes):
        for s in range(slots):

            e = np.random.uniform(0.002, 0.01)
            raan = np.radians(p * raan_spacing) + np.radians(np.random.uniform(-5, 5))
            argp = np.radians(np.random.uniform(0, 360))
            M0 = np.radians(s * M_spacing) + np.radians(np.random.uniform(-10, 10))

            sats.append({
                "const": "BDS",
                "a": a,
                "inc": inc,
                "e": e,
                "raan": raan,
                "argp": argp,
                "M0": M0,
                "color": "tab:red"
            })
    return sats



# ==========================================================
# MAIN SIMULATION WITH TIME SLIDER + TRAIL
# ==========================================================
def simulate_multiconstellation_with_slider(
        lat, lon, 
        duration_hours=4,
        dt=30,
        trail_minutes=15
    ):

    sats = gps_constellation() + galileo_constellation() + beidou_meo_constellation()

    # Time array
    T = duration_hours * 3600
    times = np.arange(0, T, dt)
    Nt = len(times)

    # Trail steps
    trail_steps = int((trail_minutes * 60) / dt)

    # =====================================================
    # Precompute all satellite positions (az, el)
    # =====================================================
    sat_positions = {}

    for sid, s in enumerate(sats):
        a, inc, raan, argp, M0, e = s["a"], s["inc"], s["raan"], s["argp"], s["M0"], s["e"]
        color = s["color"]

        n = np.sqrt(MU_EARTH / a**3)

        az_arr = np.full(Nt, np.nan)
        el_arr = np.full(Nt, np.nan)

        for k, t in enumerate(times):
            M = M0 + n * t
            nu = true_anomaly(M)

            r_eci = orbital_to_eci(a, inc, raan, argp, nu, e)
            r_ecef = eci_to_ecef(r_eci, t)

            az, el = ecef_to_az_el(r_ecef, lat, lon)

            if el > 0:  # above horizon
                az_arr[k] = az
                el_arr[k] = el

        sat_positions[sid] = {
            "color": color,
            "az": az_arr,
            "el": el_arr
        }

    # =====================================================
    # SETUP FIGURE
    # =====================================================
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)
    ax.grid(True)
    ax.set_title("Nominal GNSS Skyplot – Interactive")

    # Slider axis
    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(slider_ax, "Time Index", 0, Nt - 1, valinit=0, valstep=1)

    # =====================================================
    # INITIALIZE DRAWING OBJECTS
    # =====================================================
    sat_dots = {}   # Current position marker
    sat_tails = {}  # Trailing history

    for sid, sdat in sat_positions.items():
        color = sdat["color"]

        dot, = ax.plot([], [], "o", color=color, markersize=7)
        tail, = ax.plot([], [], "-", color=color, alpha=0.7)

        sat_dots[sid] = dot
        sat_tails[sid] = tail

    # Add legend
    ax.scatter([], [], color="tab:blue", label="GPS")
    ax.scatter([], [], color="tab:green", label="Galileo")
    ax.scatter([], [], color="tab:red", label="BeiDou MEO")
    ax.legend(loc="lower left")

    # =====================================================
    # SLIDER UPDATE FUNCTION
    # =====================================================
    def update(val):
        k = int(slider.val)

        for sid, sdat in sat_positions.items():
            az_arr = sdat["az"]
            el_arr = sdat["el"]

            # Satellite below horizon → hide
            if np.isnan(el_arr[k]):
                sat_dots[sid].set_data([], [])
                sat_tails[sid].set_data([], [])
                continue

            # ================================
            # CURRENT POSITION (scalar → list)
            # ================================
            theta = np.radians(az_arr[k])
            r = 90 - el_arr[k]
            sat_dots[sid].set_data([theta], [r])  # <-- FIX HERE

            # ================================
            # TRAILING HISTORY
            # ================================
            k0 = max(0, k - trail_steps)
            az_tail = az_arr[k0:k+1]
            el_tail = el_arr[k0:k+1]

            mask = ~np.isnan(el_tail)
            theta_tail = np.radians(az_tail[mask])
            r_tail = 90 - el_tail[mask]

            # If only 1 point in trail → wrap in list too
            if theta_tail.size == 1:
                sat_tails[sid].set_data([theta_tail[0]], [r_tail[0]])
            else:
                sat_tails[sid].set_data(theta_tail, r_tail)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()
    return sat_positions, times
    
# =====================================================================
# BUILDING SHADOW ANALYSIS MODULE
# Integrated into nominal orbit propagation script
# =====================================================================

def building_shadow_angle(L, X):
    """
    Returns the building's angular shadow width (deg)
    L = building width (meters)
    X = distance from building (meters)
    """
    return np.degrees(2 * np.arctan(L / (2 * X)))


def los_to_nlos_transitions(az_arr, el_arr, t, AZ_b, theta_block):
    """
    Identify LOS→NLOS transitions due to building shadow
    NOT horizon elevation loss, only building-blocking events.
    """
    # wrap azimuth to 0–360
    az = np.mod(az_arr, 360)

    # Visible AND not in shadow
    los = (
        (el_arr > 0) &
        ((az < AZ_b) | (az > AZ_b + theta_block))
    )

    # Visible AND inside shadow
    nlos = (
        (el_arr > 0) &
        (az >= AZ_b) &
        (az <= AZ_b + theta_block)
    )

    transition_times = []
    for k in range(len(t) - 1):
        if los[k] and nlos[k+1]:
            transition_times.append(t[k+1])

    return transition_times


def mean_gap_for_sat(az_arr, el_arr, t, AZ_b, theta_block):
    """
    Compute average time between LOS→NLOS transitions for one satellite.
    """
    ts = los_to_nlos_transitions(az_arr, el_arr, t, AZ_b, theta_block)

    if len(ts) < 2:
        return np.nan

    gaps = np.diff(ts)
    return np.mean(gaps)


def mean_gap_all_sats(sat_positions, times, L, X, AZ_b=0):
    """
    Mean LOS→NLOS time gap across all satellites.
    sat_positions[sid] must contain "az" and "el"

    L = building width (m)
    X = distance from building (m)
    AZ_b = building center azimuth (deg)
    """
    theta_block = building_shadow_angle(L, X)
    gaps = []

    for sid, sdat in sat_positions.items():
        az = sdat["az"]
        el = sdat["el"]
        g = mean_gap_for_sat(az, el, times, AZ_b, theta_block)
        if not np.isnan(g):
            gaps.append(g)

    if len(gaps) == 0:
        return np.nan, theta_block

    return np.mean(gaps), theta_block


# ---------------------------------------------------------
# Optional parameter sweeps
# ---------------------------------------------------------

def sweep_vs_distance(sat_positions, times, L_fixed, X_values, AZ_b=0):
    results = []
    for X in X_values:
        g, _ = mean_gap_all_sats(sat_positions, times, L_fixed, X, AZ_b)
        results.append(g)
    return np.array(results)


def sweep_vs_width(sat_positions, times, X_fixed, L_values, AZ_b=0):
    results = []
    for L in L_values:
        g, _ = mean_gap_all_sats(sat_positions, times, L, X_fixed, AZ_b)
        results.append(g)
    return np.array(results)

def plot_los_nlos_timeseries(sat_positions, times, L, X, AZ_b=0):
    """
    Plot a time-series showing LOS vs NLOS for every satellite.
    Uses your existing building-shadow model.
    """

    # building shadow width
    theta_block = building_shadow_angle(L, X)

    sats = sorted(sat_positions.keys())
    Ns = len(sats)

    # Build LOS/NLOS mask for each satellite
    los_mask = {}

    for sid in sats:
        az = np.mod(sat_positions[sid]["az"], 360)
        el = sat_positions[sid]["el"]

        # visible & NOT in shadow
        los = (
            (el > 0) &
            ((az < AZ_b) | (az > AZ_b + theta_block))
        )

        # visible & in shadow
        nlos = (
            (el > 0) &
            (az >= AZ_b) & (az <= AZ_b + theta_block)
        )

        # below horizon also NLOS
        los_mask[sid] = los.astype(int) - nlos.astype(int)
        # Meaning: +1 = LOS, -1 = NLOS, 0 = below horizon

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, sid in enumerate(sats):
        y = np.full(len(times), i)

        state = los_mask[sid]

        # build masked LOS and NLOS segments
        los_seg  = np.where(state == 1, i, np.nan)
        nlos_seg = np.where(state == -1, i, np.nan)

        ax.plot(times, los_seg,  color="green", linewidth=3)
        ax.plot(times, nlos_seg, color="red",   linewidth=3)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Satellite ID (index)")

    yticks = list(range(len(sats)))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"Sat {s}" for s in sats])

    ax.grid(True)
    ax.set_title(f"LOS (green) vs NLOS (red)\nBuilding at AZ={AZ_b}°, width={theta_block:.1f}° (Hoover Tower Accurate)")

    plt.tight_layout()
    plt.show()


def surface_plot_gap_times(
        sat_positions, times,
        L_values, X_values,
        AZ_b=0,
        plot_type="surface"   # or "heatmap"
    ):
    """
    Create a 3D surface (or 2D heatmap) showing:
        T_avg = mean LOS→NLOS gap time (sec)
        as function of building width L (m) and distance X (m)

    L_values: list or array of building widths
    X_values: list or array of distances from building
    """

    L_grid, X_grid = np.meshgrid(L_values, X_values)
    T_grid = np.zeros_like(L_grid, dtype=float)

    # Compute mean gap times over parameter sweep
    for i in range(len(X_values)):
        for j in range(len(L_values)):
            L = L_grid[i, j]
            X = X_grid[i, j]
            T_avg, _ = mean_gap_all_sats(sat_positions, times, L, X, AZ_b)
            T_grid[i, j] = T_avg if not np.isnan(T_avg) else np.nan

    # ======================================================
    # 3D SURFACE PLOT
    # ======================================================
    if plot_type == "surface":
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            L_grid, X_grid, T_grid,
            cmap=cm.viridis,
            edgecolor="none"
        )

        fig.colorbar(surf, shrink=0.5, aspect=10, label="Mean Gap Time (sec)")

        ax.set_xlabel("Building Width L (m)")
        ax.set_ylabel("Distance X (m)")
        ax.set_zlabel("Mean LOS→NLOS Gap (sec)")
        ax.set_title("Mean Shadow Gap Time as Function of Building Geometry")

        plt.tight_layout()
        plt.show()

    # ======================================================
    # 2D HEATMAP
    # ======================================================
    elif plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))

        c = ax.pcolormesh(L_grid, X_grid, T_grid, shading="auto", cmap="viridis")
        plt.colorbar(c, label="Mean Gap Time (sec)")

        ax.set_xlabel("Building Width L (m)")
        ax.set_ylabel("Distance X (m)")
        ax.set_title("Mean LOS→NLOS Gap Time Heatmap")

        plt.tight_layout()
        plt.show()

    return L_grid, X_grid, T_grid

# ==========================================================
# EXAMPLE RUN
# ==========================================================
if __name__ == "__main__":
    # Stanford University coordinates
    sat_positions, times = simulate_multiconstellation_with_slider(
        lat=37.4275,
        lon=-122.1697,
        duration_hours=24,
        dt=30,
        trail_minutes=15
    )
    
    ###################################
    # Following lines are for one kind of plot, everything after the next set of hashes is for a different kind of plot
    #for sid, s in sat_positions.items():
        #if np.any(~np.isnan(s["az"])):
            #az_valid = s["az"][~np.isnan(s["az"])]
            #print(f"Sat {sid}: az range = {az_valid.min():.1f} → {az_valid.max():.1f}")


    # L = 38.1       # building width in meters
    # X = 50       # distance from building in meters
    # AZ_b = 0     # building pointing North

   # T_avg, theta_block = mean_gap_all_sats(sat_positions, times, L, X, AZ_b)

    # print("Building shadow angle (deg):", theta_block)
    # print("Average LOS→NLOS gap time (sec):", T_avg)
    
    # plot_los_nlos_timeseries(sat_positions, times, L, X, AZ_b)
    ###################################
    
    L_vals = np.linspace(5, 80, 20)   # building width in meters
    X_vals = np.linspace(10, 200, 20) # distance from building

    surface_plot_gap_times(
        sat_positions, times,
        L_values=L_vals,
        X_values=X_vals,
        AZ_b=0,              # building centered at North
        plot_type="surface"  # or "heatmap"
    )


