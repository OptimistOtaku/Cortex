"""Where a memory was made. Simulator metres (x east, z south) <-> geo points Qdrant can index and filter.

Qdrant computes geo distances on Earth's radius, so metres are mapped with Earth's metres-per-degree around the
origin: a GeoRadius of 30 m is then exactly 30 simulator metres. The origin is Jezero crater (Perseverance's landing
site); the lat/lon values are an index, not survey-grade Mars coordinates.
"""

import math

ORIGIN_LAT, ORIGIN_LON = 18.4447, 77.4508
M_PER_DEG = 111_195.0  # Earth, matching Qdrant's haversine
_COS = math.cos(math.radians(ORIGIN_LAT))
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def to_geo(x, z):
    return {"lat": ORIGIN_LAT - z / M_PER_DEG, "lon": ORIGIN_LON + x / (M_PER_DEG * _COS)}


def parse_pos(text):
    """'x,z' (metres) -> {"x", "z"}; None if absent or malformed."""
    try:
        x, z = (float(v) for v in str(text).split(",")[:2])
        return {"x": round(x, 2), "z": round(z, 2)}
    except (TypeError, ValueError):
        return None


def where(pos, origin):
    """'38 m NE' of pos as seen from origin (both {"x", "z"}); '' if either is unknown."""
    if not pos or not origin:
        return ""
    dx, dn = pos["x"] - origin["x"], origin["z"] - pos["z"]  # north is -z
    d = math.hypot(dx, dn)
    if d < 3:
        return "right here"
    return f"{d:.0f} m {COMPASS[round(math.degrees(math.atan2(dx, dn)) % 360 / 45) % 8]}"
