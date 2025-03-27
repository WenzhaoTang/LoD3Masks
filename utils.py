import os
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.errors import GEOSException
from PIL import Image, ImageDraw

def parse_poslist(poslist_str, shift):
    coords = list(map(float, poslist_str.strip().split()))
    points_3d = np.array(coords).reshape(-1, 3)
    return points_3d - shift

def extract_polygon_exterior_3d(poly_elem, ns, shift):
    ext = poly_elem.find("gml:exterior", ns)
    if ext is None:
        return None
    linring = ext.find("gml:LinearRing", ns)
    if linring is None:
        return None
    pos_elem = linring.find("gml:posList", ns)
    if pos_elem is None or not pos_elem.text.strip():
        return None
    pts_3d = parse_poslist(pos_elem.text, shift)
    if pts_3d.shape[0] < 3:
        return None
    return pts_3d

def extract_polygon_interior_3d(poly_elem, ns, shift):
    interiors = []
    for interior in poly_elem.findall("gml:interior", ns):
        linring = interior.find("gml:LinearRing", ns)
        if linring is None:
            continue
        pos_elem = linring.find("gml:posList", ns)
        if pos_elem is None or not pos_elem.text.strip():
            continue
        pts_3d = parse_poslist(pos_elem.text, shift)
        if pts_3d is not None and len(pts_3d) >= 3:
            interiors.append(pts_3d)
    return interiors

def extract_all_polygons_recursively(elem, ns, shift):
    polygons_3d = []
    poly_elems = elem.findall(".//gml:Polygon", ns)
    for p in poly_elems:
        ext_3d = extract_polygon_exterior_3d(p, ns, shift)
        if ext_3d is not None and len(ext_3d) >= 3:
            polygons_3d.append(ext_3d)
    return polygons_3d

def pca_project(points_3d):
    mean_3d = np.mean(points_3d, axis=0)
    centered = points_3d - mean_3d
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:2].T
    projected_2d = centered.dot(basis)
    return projected_2d, mean_3d, basis

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
    color_list = [50, 100, 150, 200, 255] # can be manually defined
    for i, poly_2d in enumerate(polygons_2d):
        if poly_2d is None or len(poly_2d) < 3:
            continue
        px = (poly_2d - min_xy_val) * scale
        px_list = [tuple(p) for p in px]
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
            if safe_intersects(fixed_polys[i], fixed_polys[j]) or (safe_distance(fixed_polys[i], fixed_polys[j]) < dist_thresh):
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
    return [g for g in grouped_polys if g and not g.is_empty and g.area > 0]

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
