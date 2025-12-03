import folium
import numpy as np


def intersect_ray_with_segment(P0, d, A, B):
    """
    Compute intersection of ray P(t) = P0 + t*d, t >= 0
    with segment A -> B in (lon, lat) coordinates.
    """
    P0 = np.array(P0, float)
    d = np.array(d, float)
    A = np.array(A, float)
    B = np.array(B, float)
    s = B - A

    M = np.array([[d[0], -s[0]],
                  [d[1], -s[1]]])

    try:
        t, u = np.linalg.solve(M, A - P0)
    except np.linalg.LinAlgError:
        return None

    if t >= 0 and 0 <= u <= 1:
        return t
    return None


def create_interactive_map(center_lat, center_lon, polygon_pts):
    """
    center_lat, center_lon : central reference point
    polygon_pts : list of four (lat, lon, alt) tuples (alt ignored)

    Returns a Folium map.
    """

    # Ignore altitude values
    poly2d = [(lat, lon) for (lat, lon, _) in polygon_pts]

    # Create zoomed-in map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

    # Draw polygon
    folium.PolyLine(poly2d + [poly2d[0]], color="red", weight=3).add_to(m)

    # Draw center point as black star
    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.Icon(color="black", icon="star", prefix="fa")
    ).add_to(m)

    # Build polygon edges
    edges = []
    for i in range(len(poly2d)):
        edges.append((poly2d[i], poly2d[(i + 1) % len(poly2d)]))

    # Spoke angles (degrees clockwise from north)
    angles_deg = np.arange(0, 289, 72)

    # Center in (lon, lat)
    P0 = (center_lon, center_lat)

    for angle in angles_deg:
        theta = np.deg2rad(angle)
        d = np.array([np.sin(theta), np.cos(theta)])  # (Δlon, Δlat)

        best_t = None
        best_pt = None

        for A, B in edges:
            A_xy = (A[1], A[0])  # (lon, lat)
            B_xy = (B[1], B[0])

            t = intersect_ray_with_segment(P0, d, A_xy, B_xy)
            if t is not None:
                if best_t is None or t < best_t:
                    best_t = t
                    best_pt = P0 + t * d

        if best_pt is not None:
            end_lon, end_lat = best_pt

            # Draw dashed line
            folium.PolyLine(
                [(center_lat, center_lon), (end_lat, end_lon)],
                color="blue",
                weight=2,
                dash_array="5,10"
            ).add_to(m)

            # Label position: slightly before the end point (90% of t)
            label_lon = P0[0] + 0.9 * best_t * d[0]
            label_lat = P0[1] + 0.9 * best_t * d[1]

            # Heading label
            label_text = f"{angle}°"
            folium.Marker(
                location=[label_lat, label_lon],
                icon=folium.DivIcon(
                    html=f"""
                        <div style="font-size:20px; 
                                    font-weight:bold; 
                                    color:black;
                                    text-shadow:1px 1px 2px white;">
                            {label_text}
                        </div>
                    """
                )
            ).add_to(m)

    return m


if __name__ == "__main__":
    center = (37.4275, -122.17)

    # Stanford main quad area dimensions in lat/lon (source: Google maps).
    # Note: We are assuming the receiver and quad have the same altitude.
    N_corner = (37.42800462101945, -122.17110561760701, 0)
    W_corner = (37.427368467502745, -122.17132167055206, 1)
    E_corner = (37.427590917103906, -122.16917077508404, 4)
    S_corner = (37.42695582911489, -122.16938586988063, 5)

    # Define a polygon with vertices at these coordinates.
    polygon_lla = [
        N_corner, E_corner, S_corner, W_corner
    ]

    m = create_interactive_map(center[0], center[1], polygon_lla)

    file_path = "my_map.html"
    m.save(file_path)

