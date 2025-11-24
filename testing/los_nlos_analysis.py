# ==========================================================
# gnss_core.py
# Core GNSS orbit propagation + LOS/NLOS building occlusion
# ==========================================================

import numpy as np

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
# Kepler equation (simplified for nominal MEO e~small)
# ==========================================================
def true_anomaly(M):
    return M


# ==========================================================
# Orbit → ECI
# ==========================================================
def orbital_to_eci(a, inc, raan, argp, nu, e):
    r = a * (1 - e*e) / (1 + e*np.cos(nu))
    r_orb = np.array([r, 0, 0])
    u = argp + nu
    return Rz(raan) @ (Rx(inc) @ (Rz(u) @ r_orb))


# ==========================================================
# ECI → ECEF
# ==========================================================
def eci_to_ecef(r_eci, t):
    return Rz(-OMEGA_EARTH * t) @ r_eci


# ==========================================================
# ECEF → az/el
# ==========================================================
def ecef_to_az_el(r_ecef, lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

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
# Constellations
# ==========================================================
def gps_constellation():
    sats = []
    a = R_EARTH + 20200e3
    inc = np.radians(55)
    planes = 6
    slots = 4

    for p in range(planes):
        for s in range(slots):
            sats.append({
                "const": "GPS",
                "a": a,
                "inc": inc,
                "e": np.random.uniform(0.005, 0.02),
                "raan": np.radians(60*p + np.random.uniform(-3, 3)),
                "argp": np.radians(np.random.uniform(0, 360)),
                "M0": np.radians(90*s + np.random.uniform(-5, 5)),
                "color": "tab:blue"
            })
    return sats


def galileo_constellation():
    sats = []
    a = R_EARTH + 23222e3
    inc = np.radians(56)
    planes = 3
    slots = 8

    for p in range(planes):
        for s in range(slots):
            sats.append({
                "const": "GAL",
                "a": a,
                "inc": inc,
                "e": np.random.uniform(0.0005, 0.005),
                "raan": np.radians(120*p + np.random.uniform(-5, 5)),
                "argp": np.radians(np.random.uniform(0, 360)),
                "M0": np.radians(45*s + np.random.uniform(-10, 10)),
                "color": "tab:green"
            })
    return sats


def beidou_meo_constellation():
    sats = []
    a = R_EARTH + 27878e3
    inc = np.radians(55)
    planes = 3
    slots = 8

    for p in range(planes):
        for s in range(slots):
            sats.append({
                "const": "BDS",
                "a": a,
                "inc": inc,
                "e": np.random.uniform(0.002, 0.01),
                "raan": np.radians(120*p + np.random.uniform(-5, 5)),
                "argp": np.radians(np.random.uniform(0, 360)),
                "M0": np.radians(45*s + np.random.uniform(-10, 10)),
                "color": "tab:red"
            })
    return sats


# ==========================================================
# Master constellation simulator
# ==========================================================
def simulate_constellations(lat, lon, duration_hours=12, dt=30):
    sats = gps_constellation() + galileo_constellation() + beidou_meo_constellation()

    T = duration_hours * 3600
    times = np.arange(0, T, dt)
    Nt = len(times)

    sat_positions = {}

    for sid, s in enumerate(sats):
        a, inc, raan, argp, M0, e = s["a"], s["inc"], s["raan"], s["argp"], s["M0"], s["e"]
        color = s["color"]

        n = np.sqrt(MU_EARTH / a**3)
        az_arr = np.full(Nt, np.nan)
        el_arr = np.full(Nt, np.nan)

        for k, t in enumerate(times):
            M = M0 + n*t
            nu = true_anomaly(M)
            r_eci = orbital_to_eci(a, inc, raan, argp, nu, e)
            r_ecef = eci_to_ecef(r_eci, t)
            az, el = ecef_to_az_el(r_ecef, lat, lon)

            if el > 0:
                az_arr[k] = az
                el_arr[k] = el

        sat_positions[sid] = {"color": color, "az": az_arr, "el": el_arr}

    return sat_positions, times


# ==========================================================
# Building-shadow occlusion model
# ==========================================================
def building_shadow_angle(L, X):
    return np.degrees(2 * np.arctan(L / (2 * X)))

def building_elevation_mask(H, X):
    return np.degrees(np.arctan(H / X))


def los_nlos_mask(az_arr, el_arr, AZ_b, theta_block, elev_cut):
    az = np.mod(az_arr, 360)
    el = el_arr

    inside_az = (az >= AZ_b) & (az <= AZ_b + theta_block)
    inside_el = (el > 0) & (el <= elev_cut)

    nlos = inside_az & inside_el
    los = (el > 0) & (~nlos)

    return np.where(los, 1, np.where(nlos, -1, 0))


def global_shadow_transition_times(sat_positions, times, L, X, H, AZ_b=0):
    theta_block = building_shadow_angle(L, X)
    elev_cut = building_elevation_mask(H, X)

    events = []

    for sdat in sat_positions.values():
        az = sdat["az"]
        el = sdat["el"]
        state = los_nlos_mask(az, el, AZ_b, theta_block, elev_cut)

        for k in range(len(times) - 1):
            if state[k] != state[k+1] and 0 not in (state[k], state[k+1]):
                events.append(times[k+1])

    return np.array(sorted(events)), theta_block, elev_cut


def global_mean_event_gap(sat_positions, times, L, X, H, AZ_b=0):
    events, _, _ = global_shadow_transition_times(sat_positions, times, L, X, H, AZ_b)
    if len(events) < 2:
        return np.nan
    return np.mean(np.diff(events))
