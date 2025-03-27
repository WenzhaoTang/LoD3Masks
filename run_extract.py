import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.ops import unary_union
from shapely.errors import GEOSException
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import (
    parse_poslist,
    extract_polygon_exterior_3d,
    extract_polygon_interior_3d,
    extract_all_polygons_recursively,
    pca_project,
    to_shapely_polygon,
    safe_buffer0,
    safe_intersects,
    safe_distance,
    debug_save_multi_polygon_2d,
    group_polygons_by_proximity,
    polygon_or_multipoly_to_2dcoords
)

# Conventional Translation(Used in Scan2LoD3)
SHIFT = np.array([690729.0567, 5335897.0603, 500.0])

# XML namespaces
ns = {
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml'
}

file_path = os.path.join("data", "input.gml")
output_dir = os.path.join("output", "mask_extraction")
os.makedirs(output_dir, exist_ok=True)
debug_dir = os.path.join(output_dir, "debug_masks")
os.makedirs(debug_dir, exist_ok=True)

tree = ET.parse(file_path)
root = tree.getroot()

def process_wall(xml_bytes):
    wall = ET.fromstring(xml_bytes)
    wall_id = wall.attrib.get("{http://www.opengis.net/gml}id", "unknown")
    print(f"\n=== Processing WallSurface: {wall_id} ===")
    
    facade_polygons_3d = extract_all_polygons_recursively(wall, ns, SHIFT)
    if not facade_polygons_3d:
        print(f"No facade polygons found, skipping {wall_id}")
        return wall_id

    opening_polygons_3d = []
    for poly_elem in wall.findall(".//gml:Polygon", ns):
        int_3d_list = extract_polygon_interior_3d(poly_elem, ns, SHIFT)
        if int_3d_list:
            opening_polygons_3d.extend(int_3d_list)

    for elem in wall.findall(".//bldg:opening", ns):
        opening_polygons_3d.extend(extract_all_polygons_recursively(elem, ns, SHIFT))
    for elem in wall.findall(".//bldg:Door", ns):
        opening_polygons_3d.extend(extract_all_polygons_recursively(elem, ns, SHIFT))
    for elem in wall.findall(".//bldg:Window", ns):
        opening_polygons_3d.extend(extract_all_polygons_recursively(elem, ns, SHIFT))

    if not opening_polygons_3d:
        print(f"No openings found in {wall_id}, skipping hole generation.")

    all_points_3d = facade_polygons_3d if not opening_polygons_3d else (facade_polygons_3d + opening_polygons_3d)
    all_points_3d_np = np.vstack(all_points_3d)
    projected_all, mean_3d, basis_3d = pca_project(all_points_3d_np)
    project_3d = lambda pts: (pts - mean_3d).dot(basis_3d)

    debug_facade_2d = [project_3d(f3d) for f3d in facade_polygons_3d]
    debug_opening_2d = [project_3d(o3d) for o3d in opening_polygons_3d]
    debug_path = os.path.join(debug_dir, f"debug_{wall_id}_all_polygons.png")
    debug_save_multi_polygon_2d(debug_facade_2d + debug_opening_2d, debug_path, scale=100)

    facade_shapely = []
    for f3d in facade_polygons_3d:
        f2d = project_3d(f3d)
        poly = safe_buffer0(to_shapely_polygon(f2d))
        if poly and not poly.is_empty:
            facade_shapely.append(poly)
    if not facade_shapely:
        print(f"Invalid facade polygons after projection, skipping {wall_id}")
        return wall_id

    try:
        facade_polygon = safe_buffer0(unary_union(facade_shapely))
    except GEOSException as e:
        print(f"[Warning] Union of facade polygons failed: {e}")
        return wall_id

    if not facade_polygon or facade_polygon.is_empty:
        print(f"Facade polygon is empty after union, skipping {wall_id}")
        return wall_id

    opening_shapely = []
    for op_3d in opening_polygons_3d:
        op2d = project_3d(op_3d)
        poly = safe_buffer0(to_shapely_polygon(op2d))
        if poly and not poly.is_empty:
            opening_shapely.append(poly)

    if not opening_shapely:
        print(f"No valid opening polygons for {wall_id}, outputting facade only.")
        openings_union = None
    else:
        grouped = group_polygons_by_proximity(opening_shapely, dist_thresh=0.1)
        if grouped:
            debug_group = []
            for g in grouped:
                if g.geom_type == "Polygon":
                    debug_group.append(np.array(g.exterior.coords))
                elif g.geom_type == "MultiPolygon":
                    for subg in g.geoms:
                        debug_group.append(np.array(subg.exterior.coords))
            debug_group_path = os.path.join(debug_dir, f"debug_{wall_id}_grouped_openings.png")
            debug_save_multi_polygon_2d(debug_group, debug_group_path, scale=100)
            try:
                openings_union = safe_buffer0(unary_union(grouped))
            except Exception as e:
                print(f"[Warning] Union of grouped openings failed: {e}")
                openings_union = None
        else:
            print(f"No grouped openings after proximity, skipping {wall_id}")
            openings_union = None

    if openings_union and not openings_union.is_empty:
        try:
            facade_with_holes = safe_buffer0(facade_polygon.difference(openings_union))
        except GEOSException as e:
            print(f"[Warning] Difference operation failed: {e}")
            facade_with_holes = facade_polygon
    else:
        facade_with_holes = facade_polygon

    if not facade_with_holes or facade_with_holes.is_empty:
        print(f"Facade with holes is empty, skipping {wall_id}")
        return wall_id

    if facade_with_holes.geom_type == "Polygon":
        polygons_to_export = [facade_with_holes]
    elif facade_with_holes.geom_type == "MultiPolygon":
        polygons_to_export = list(facade_with_holes.geoms)
    else:
        polygons_to_export = []

    all_rings = []
    for poly in polygons_to_export:
        if poly.exterior is not None:
            all_rings.append(np.array(poly.exterior.coords))
        for ring in poly.interiors:
            all_rings.append(np.array(ring.coords))
    if not all_rings:
        print(f"No exterior ring in final geometry, skipping {wall_id}")
        return wall_id
    all_rings = np.vstack(all_rings)
    min_xy = np.min(all_rings, axis=0)
    max_xy = np.max(all_rings, axis=0)
    scale_factor = 100
    width = int((max_xy[0] - min_xy[0]) * scale_factor)
    height = int((max_xy[1] - min_xy[1]) * scale_factor)
    if width < 1 or height < 1:
        print(f"Invalid image size, skipping {wall_id}")
        return wall_id

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    def ring_to_pixels(coords):
        return [tuple(p) for p in ((coords - min_xy) * scale_factor)]
    for poly in polygons_to_export:
        if poly.exterior is not None:
            draw.polygon(ring_to_pixels(np.array(poly.exterior.coords)), fill=255)
        for ring in poly.interiors:
            draw.polygon(ring_to_pixels(np.array(ring.coords)), fill=0)
    out_path = os.path.join(output_dir, f"mask_{wall_id}.png")
    mask.save(out_path)
    print(f"Saved final mask for wall {wall_id} to {out_path}")
    return wall_id

def main():
    walls = root.findall(".//bldg:WallSurface", ns)
    xml_list = [ET.tostring(w) for w in walls]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [exe.submit(process_wall, xml) for xml in xml_list]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Walls"):
            pass

if __name__ == '__main__':
    main()
