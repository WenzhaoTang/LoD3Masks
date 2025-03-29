import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.errors import GEOSException

DEBUG = True
MASK_TYPE = "all" # "all", "door", "window", "full"

# Convention from Scan2LoD3
SHIFT = np.array([690729.0567, 5335897.0603, 500.0])
ns = {
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml'
}

def parse_poslist(poslist_str):
    coords = list(map(float, poslist_str.strip().split()))
    points_3d = np.array(coords).reshape(-1, 3)
    return points_3d - SHIFT

def extract_polygon_exterior_3d(poly_elem):
    ext = poly_elem.find("gml:exterior", ns)
    if ext is None:
        return None
    lr = ext.find("gml:LinearRing", ns)
    if lr is None:
        return None
    pos_elem = lr.find("gml:posList", ns)
    if pos_elem is None or not pos_elem.text.strip():
        return None
    pts = parse_poslist(pos_elem.text)
    if pts.shape[0] < 3:
        return None
    return pts

def extract_polygon_interior_3d(poly_elem):
    interiors = []
    for interior in poly_elem.findall("gml:interior", ns):
        lr = interior.find("gml:LinearRing", ns)
        if lr is None:
            continue
        pos_elem = lr.find("gml:posList", ns)
        if pos_elem is None or not pos_elem.text.strip():
            continue
        pts = parse_poslist(pos_elem.text)
        if pts is not None and len(pts) >= 3:
            interiors.append(pts)
    return interiors

def extract_all_polygons_recursively(elem, ns):
    polygons = []
    for p in elem.findall(".//gml:Polygon", ns):
        pts = extract_polygon_exterior_3d(p)
        if pts is not None and len(pts) >= 3:
            polygons.append(pts)
    return polygons

def to_shapely_polygon(points_2d):
    if len(points_2d) < 3:
        return None
    return Polygon(points_2d)

def safe_buffer0(geom):
    if geom is None or geom.is_empty:
        return geom
    try:
        fixed = geom.buffer(0)
        return fixed if not fixed.is_empty else None
    except GEOSException:
        return None

def safe_intersects(a, b):
    if a is None or b is None:
        return False
    try:
        return a.intersects(b)
    except GEOSException:
        return False

def safe_distance(a, b):
    if a is None or b is None:
        return 999999999
    try:
        return a.distance(b)
    except GEOSException:
        return 999999999

def debug_save_multi_polygon_2d(polygons_2d, out_path, scale=100):
    if not polygons_2d:
        return
    all_pts = [pts for pts in polygons_2d if pts is not None and len(pts) >= 3]
    if not all_pts:
        return
    all_pts_np = np.vstack(all_pts)
    min_xy_val = np.min(all_pts_np, axis=0)
    max_xy_val = np.max(all_pts_np, axis=0)
    width = int((max_xy_val[0] - min_xy_val[0]) * scale)
    height = int((max_xy_val[1] - min_xy_val[1]) * scale)
    if width < 2 or height < 2:
        return
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    color_list = [50, 100, 150, 200, 255]
    for i, poly_2d in enumerate(polygons_2d):
        if poly_2d is None or len(poly_2d) < 3:
            continue
        px = (poly_2d - min_xy_val) * scale
        px_flipped = np.column_stack((width - px[:, 0], height - px[:, 1]))
        px_list = [tuple(p) for p in px_flipped]
        c = color_list[i % len(color_list)]
        draw.polygon(px_list, fill=c)
    img.save(out_path)

def compute_polygon_normal(points_3d):
    if len(points_3d) < 3:
        return None
    p0, p1, p2 = points_3d[0], points_3d[1], points_3d[2]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-8:
        return None
    return normal / nlen

def define_axes_from_normal(facade_normal):
    up = np.array([0, 0, 1], dtype=float)
    dot_val = np.dot(up, facade_normal)
    up_proj = up - dot_val * facade_normal
    length = np.linalg.norm(up_proj)
    if length < 1e-8:
        return None
    up_proj /= length
    x_axis = np.cross(facade_normal, up_proj)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = up_proj
    return x_axis, y_axis

def project_points(points_3d, x_axis, y_axis):
    return np.column_stack((points_3d.dot(x_axis), points_3d.dot(y_axis)))
