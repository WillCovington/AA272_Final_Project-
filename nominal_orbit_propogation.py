import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# ==========================================================
# EXAMPLE RUN
# ==========================================================
if __name__ == "__main__":
    # Stanford University coordinates
    simulate_multiconstellation_with_slider(
        lat=37.4275,
        lon=-122.1697,
        duration_hours=4,
        dt=30,
        trail_minutes=15
    )
