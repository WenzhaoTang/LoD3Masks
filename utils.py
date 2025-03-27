import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.errors import GEOSException

SHIFT = np.array([690729.0567, 5335897.0603, 500.0])
ns = {
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml"
}

def parse_poslist(poslist_str):
    coords = list(map(float, poslist_str.strip().split()))
    points_3d = np.array(coords).reshape(-1, 3)
    return points_3d - SHIFT

def extract_polygon_exterior_3d(poly_elem):
    ext = poly_elem.find("gml:exterior", ns)
    if ext is None:
        return None
    linring = ext.find("gml:LinearRing", ns)
    if linring is None:
        return None
    pos_elem = linring.find("gml:posList", ns)
    if pos_elem is None or not pos_elem.text.strip():
        return None
    pts_3d = parse_poslist(pos_elem.text)
    if pts_3d.shape[0] < 3:
        return None
    return pts_3d

def extract_polygon_interior_3d(poly_elem):
    interiors = []
    for interior in poly_elem.findall("gml:interior", ns):
        linring = interior.find("gml:LinearRing", ns)
        if linring is None:
            continue
        pos_elem = linring.find("gml:posList", ns)
        if pos_elem is None or not pos_elem.text.strip():
            continue
        pts_3d = parse_poslist(pos_elem.text)
        if pts_3d is not None and len(pts_3d) >= 3:
            interiors.append(pts_3d)
    return interiors

def extract_all_polygons_recursively(elem, ns):
    polygons_3d = []
    poly_elems = elem.findall(".//gml:Polygon", ns)
    for p in poly_elems:
        ext_3d = extract_polygon_exterior_3d(p)
        if ext_3d is not None and len(ext_3d) >= 3:
            polygons_3d.append(ext_3d)
    return polygons_3d

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
    all_pts = []
    for pts in polygons_2d:
        if pts is not None and len(pts) >= 3:
            all_pts.append(pts)
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

def group_polygons_by_proximity(polygons, dist_thresh=0.1):
    fixed_polys = []
    for p in polygons:
        fp = safe_buffer0(p)
        if fp and not fp.is_empty and fp.area > 0:
            fixed_polys.append(fp)
    n = len(fixed_polys)
    if n == 0:
        return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            pi = fixed_polys[i]
            pj = fixed_polys[j]
            if safe_intersects(pi, pj) or (safe_distance(pi, pj) < dist_thresh):
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    groups = []
    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            comp = [i]
            while queue:
                cur = queue.pop()
                for nxt in adj[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        queue.append(nxt)
                        comp.append(nxt)
            groups.append(comp)
    grouped_polys = []
    for comp in groups:
        sublist = [fixed_polys[k] for k in comp]
        try:
            unioned = unary_union(sublist)
            if not unioned.is_empty:
                grouped_polys.append(safe_buffer0(unioned))
        except GEOSException:
            pass
    result = [g for g in grouped_polys if g and not g.is_empty and g.area > 0]
    return result

def polygon_or_multipoly_to_2dcoords(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [np.array(geom.exterior.coords)]
    elif geom.geom_type == "MultiPolygon":
        arrs = []
        for g in geom.geoms:
            if g.exterior:
                arrs.append(np.array(g.exterior.coords))
        return arrs
    return []

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
    projected_2d = []
    for p in points_3d:
        x = np.dot(p, x_axis)
        y = np.dot(p, y_axis)
        projected_2d.append([x, y])
    return np.array(projected_2d)
