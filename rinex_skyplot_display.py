import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# EARTH CONSTANTS
# ============================================================
MU = 3.986005e14           # GM WGS-84
OMEGA_E = 7.2921151467e-5  # Earth rotation rate
R_EARTH = 6378137.0        # Earth radius (m)


# ============================================================
# PARSE RINEX NAV FILE (GPS ONLY)
# ============================================================
def parse_float(s):
    return float(s.replace("D", "E").replace("d", "e"))

def parse_rinex_nav_r1(path):
    sats = {}
    with open(path, "r") as f:
        lines = f.readlines()

    # Skip header
    i = 0
    while "END OF HEADER" not in lines[i]:
        i += 1
    i += 1

    # Each satellite block = 8 lines
    while i < len(lines):

        line1 = lines[i];     i += 1
        if len(line1) < 2 or not line1[0].isdigit():
            continue

        prn = int(line1[0:2])
        year  = int(line1[2:5])
        month = int(line1[5:8])
        day   = int(line1[8:11])
        hour  = int(line1[11:14])
        minute= int(line1[14:17])
        second= parse_float(line1[17:22])

        af0 = parse_float(line1[22:41])
        af1 = parse_float(line1[41:60])
        af2 = parse_float(line1[60:79])

        # Remaining lines
        line2 = lines[i]; i+=1
        line3 = lines[i]; i+=1
        line4 = lines[i]; i+=1
        line5 = lines[i]; i+=1
        line6 = lines[i]; i+=1
        line7 = lines[i]; i+=1
        line8 = lines[i]; i+=1

        # Extract orbital parameters
        IODE     = parse_float(line2[3:22])
        Crs      = parse_float(line2[22:41])
        dn       = parse_float(line2[41:60])
        M0       = parse_float(line2[60:79])

        Cuc      = parse_float(line3[3:22])
        e        = parse_float(line3[22:41])
        Cus      = parse_float(line3[41:60])
        sqrtA    = parse_float(line3[60:79])
        a        = sqrtA * sqrtA

        toe      = parse_float(line4[3:22])
        Cic      = parse_float(line4[22:41])
        OMEGA0   = parse_float(line4[41:60])
        Cis      = parse_float(line4[60:79])

        i0       = parse_float(line5[3:22])
        Crc      = parse_float(line5[22:41])
        omega    = parse_float(line5[41:60])
        OMEGADOT = parse_float(line5[60:79])

        iDOT     = parse_float(line6[3:22])

        sats[prn] = {
            "a": a,
            "e": e,
            "i0": i0,
            "omega": omega,
            "M0": M0,
            "OMEGA0": OMEGA0,
            "OMEGADOT": OMEGADOT,
            "dn": dn,
            "toe": toe,
            "Cuc": Cuc, "Cus": Cus,
            "Cic": Cic, "Cis": Cis,
            "Crc": Crc, "Crs": Crs,
            "iDOT": iDOT
        }

    return sats



# ============================================================
# KEPLER'S EQUATION SOLVER
# ============================================================
def solve_kepler(M, e, tol=1e-12):
    E = M
    for _ in range(10):
        E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    return E


# ============================================================
# ORBIT PROPAGATION (GPS BROADCAST MODEL)
# ============================================================
def propagate_satellite(t, eph):
    a = eph["a"]
    e = eph["e"]
    M0 = eph["M0"]
    dn = eph["dn"]
    toe = eph["toe"]
    Cuc = eph["Cuc"]; Cus = eph["Cus"]
    Cic = eph["Cic"]; Cis = eph["Cis"]
    Crc = eph["Crc"]; Crs = eph["Crs"]
    i0 = eph["i0"]
    OMEGA0 = eph["OMEGA0"]
    OMEGADOT = eph["OMEGADOT"]
    iDOT = eph["iDOT"]
    omega = eph["omega"]

    dt = t - toe

    # Mean motion
    n0 = np.sqrt(MU / a**3)
    n = n0 + dn

    # Mean anomaly
    M = M0 + n * dt

    # Solve Kepler
    E = solve_kepler(M, e)

    # True anomaly
    v = np.arctan2(np.sqrt(1-e**2)*np.sin(E), np.cos(E)-e)

    # Argument of latitude
    u = v + omega

    # Radius
    r = a*(1 - e*np.cos(E))

    # Corrections
    u_corr = u + Cuc*np.cos(2*u) + Cus*np.sin(2*u)
    r_corr = r + Crc*np.cos(2*u) + Crs*np.sin(2*u)
    i_corr = i0 + iDOT*dt + Cic*np.cos(2*u) + Cis*np.sin(2*u)

    # Positions in orbital plane
    x_orb = r_corr * np.cos(u_corr)
    y_orb = r_corr * np.sin(u_corr)

    # Corrected RAAN
    OMEGA = OMEGA0 + (OMEGADOT - OMEGA_E) * dt

    # ECEF coordinates
    x = x_orb * np.cos(OMEGA) - y_orb * np.cos(i_corr)*np.sin(OMEGA)
    y = x_orb * np.sin(OMEGA) + y_orb * np.cos(i_corr)*np.cos(OMEGA)
    z = y_orb * np.sin(i_corr)

    return np.array([x, y, z])


# ============================================================
# ECEF → ENU → AZ/EL
# ============================================================
def ecef_to_az_el(r_ecef, lat_deg, lon_deg, alt_m):
    # Receiver ECEF (simple spherical approx)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    R = R_EARTH + alt_m
    rx = R*np.cos(lat)*np.cos(lon)
    ry = R*np.cos(lat)*np.sin(lon)
    rz = R*np.sin(lat)
    r_rx = np.array([rx, ry, rz])

    dr = r_ecef - r_rx

    # Rotation to ENU
    Rmat = np.array([
        [-np.sin(lon),             np.cos(lon),              0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])

    e, n, u = Rmat @ dr
    az = np.degrees(np.arctan2(e, n)) % 360
    el = np.degrees(np.arctan2(u, np.sqrt(e*e + n*n)))

    return az, el


# ============================================================
# MAIN SKY PLOT SIMULATION
# ============================================================
def skyplot_from_rinex(path, lat, lon, alt, duration_hours=4, dt=30, trail_min=15):

    eph = parse_rinex_nav_r1(path)
    sats = sorted(eph.keys())

    times = np.arange(0, duration_hours*3600, dt)
    Nt = len(times)
    trail_steps = int(trail_min*60/dt)

    # Storage
    AZ = {s: np.full(Nt, np.nan) for s in sats}
    EL = {s: np.full(Nt, np.nan) for s in sats}

    # Propagate all sats
    for k, t in enumerate(times):
        for s in sats:
            r = propagate_satellite(t, eph[s])
            az, el = ecef_to_az_el(r, lat, lon, alt)
            if el > 0:
                AZ[s][k] = az
                EL[s][k] = el

    # ==========================================================
    # PLOT
    # ==========================================================
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)
    ax.grid(True)
    ax.set_title("Real Satellite Skyplot (RINEX Navigation)")

    # slider
    axsl = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(axsl, "Index", 0, Nt-1, valinit=0, valstep=1)

    dots = {}
    tails = {}

    cmap = plt.cm.get_cmap("tab20")

    for i, s in enumerate(sats):
        color = cmap(i % 20)
        dot, = ax.plot([], [], "o", color=color)
        tail, = ax.plot([], [], "-", color=color, alpha=0.6)
        dots[s] = dot
        tails[s] = tail

    def update(val):
        k = int(slider.val)
        for s in sats:
            if not np.isnan(EL[s][k]):
                theta = np.radians(AZ[s][k])
                r = 90 - EL[s][k]
                if r < 0.5:
                    r = 0.5  # never allow r = 0
                dots[s].set_data([theta], [r])

                # ------- TRAIL -------
                k0 = max(0, k - trail_steps)
                mask = ~np.isnan(EL[s][k0:k+1])
                th_tail = np.radians(AZ[s][k0:k+1][mask])
                r_tail = 90 - EL[s][k0:k+1][mask]
                r_tail[r_tail < 0.5] = 0.5

                tails[s].set_data(th_tail, r_tail)
            else:
                dots[s].set_data([], [])
                tails[s].set_data([], [])

        fig.canvas.draw_idle()


    slider.on_changed(update)
    update(0)
    plt.show()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    skyplot_from_rinex(
        path="./rinex_nav_files/brdc3230.25n",
        lat=37.4275,
        lon=-122.1697,
        alt=20,
        duration_hours=4,
        dt=30,
        trail_min=15
    )
