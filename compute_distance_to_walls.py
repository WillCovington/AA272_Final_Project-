import numpy as np
import pymap3d as pm
import matplotlib.pyplot as plt

def lla_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert LLA → ENU with respect to reference LLA (lat0, lon0, alt0)."""
    e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0, ell=None, deg=True)
    return np.array([e, n, u])


def heading_to_unit_vector(heading_deg):
    """
    Convert heading (azimuth) clockwise from north into ENU unit vector.
    ENU axes: x=East, y=North, z=Up
    """
    az = np.deg2rad(heading_deg)
    e = np.sin(az)   # East component
    n = np.cos(az)   # North component
    return np.array([e, n, 0.0])


def ray_segment_intersection(ray_origin, ray_dir, p1, p2, eps=1e-9):
    """
    Compute intersection between infinite ray and segment p1->p2 in 2D (E,N).
    Works in EN plane (z ignored).
    Returns (t, u) where
        ray:    r = ray_origin + t * ray_dir       with t >= 0
        segment: s = p1 + u*(p2 - p1)              with 0 <= u <= 1
    If no intersection exists, return None.
    """
    r0 = ray_origin[:2]
    rD = ray_dir[:2]
    s1 = p1[:2]
    sD = (p2 - p1)[:2]

    # Solve r0 + t*rD = s1 + u*sD  →  2×2 linear system
    A = np.column_stack((rD, -sD))
    b = s1 - r0

    if abs(np.linalg.det(A)) < eps:
        return None  # Ray and segment parallel or nearly so

    t, u = np.linalg.solve(A, b)

    if t >= 0 and 0 <= u <= 1:
        return t  # distance multiplier along ray
    else:
        return None


def distance_to_polygon_in_heading(polygon_lla, observer_lla, heading_deg):
    """
    polygon_lla: list of (lat, lon, alt)
    observer_lla: (lat, lon, alt)
    heading_deg: azimuth angle of observer (° clockwise from North)

    Returns the minimum distance from the observer along the heading direction
    to the polygon boundary, or None if no intersection.
    """
    lat0, lon0, alt0 = observer_lla

    # Convert observer and polygon to ENU
    obsENU = lla_to_enu(lat0, lon0, alt0, lat0, lon0, alt0)
    polyENU = [lla_to_enu(lat, lon, alt, lat0, lon0, alt0)
               for lat, lon, alt in polygon_lla]

    # Close polygon loop
    polyENU.append(polyENU[0])

    # Ray direction
    d = heading_to_unit_vector(heading_deg)

    min_dist = np.inf

    # Test intersection with each edge
    for i in range(len(polyENU)-1):
        p1 = polyENU[i]
        p2 = polyENU[i+1]

        t = ray_segment_intersection(obsENU, d, p1, p2)

        if t is not None:
            dist = t * np.linalg.norm(d)
            if dist < min_dist:
                min_dist = dist

    if np.isinf(min_dist):
        return None
    else:
        return float(min_dist)
