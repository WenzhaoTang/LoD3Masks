import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.ops import unary_union
from utils import (
    extract_polygon_exterior_3d,
    extract_polygon_interior_3d,
    extract_all_polygons_recursively,
    to_shapely_polygon,
    safe_buffer0,
    safe_intersects,
    safe_distance,
    debug_save_multi_polygon_2d,
    group_polygons_by_proximity,
    polygon_or_multipoly_to_2dcoords,
    compute_polygon_normal,
    define_axes_from_normal,
    project_points,
    ns
)

def process_wall(xml_bytes, debug_dir, output_dir):
    wall = ET.fromstring(xml_bytes)
    wall_id = wall.attrib.get("{http://www.opengis.net/gml}id", "unknown")
    print(f"\n=== Processing WallSurface: {wall_id} ===")
    facade_polygons_3d = extract_all_polygons_recursively(wall, ns)
    if not facade_polygons_3d:
        print(f"  No facade polygons found, skip {wall_id}")
        return wall_id
    opening_polygons_3d = []
    facade_poly_elems = wall.findall(".//gml:Polygon", ns)
    for poly_elem in facade_poly_elems:
        int_3d_list = extract_polygon_interior_3d(poly_elem)
        if int_3d_list:
            opening_polygons_3d.extend(int_3d_list)
    for op_elem_tag in ["bldg:opening", "bldg:Door", "bldg:Window"]:
        for op_elem in wall.findall(f".//{op_elem_tag}", ns):
            polys_3d = extract_all_polygons_recursively(op_elem, ns)
            opening_polygons_3d.extend(polys_3d)
    normal = compute_polygon_normal(facade_polygons_3d[0])
    if normal is None:
        print(f"  Invalid facade normal, skip {wall_id}")
        return wall_id
    axes = define_axes_from_normal(normal)
    if axes is None:
        print(f"  Facade is nearly horizontal, skip {wall_id}")
        return wall_id
    x_axis, y_axis = axes
    def project_3d(pts_3d):
        return project_points(pts_3d, x_axis, y_axis)
    debug_facade_2d_list = [project_3d(f3d) for f3d in facade_polygons_3d]
    debug_opening_2d_list = [project_3d(o3d) for o3d in opening_polygons_3d]
    debug_all_path = os.path.join(debug_dir, f"debug_{wall_id}_all_polygons.png")
    debug_save_multi_polygon_2d(debug_facade_2d_list + debug_opening_2d_list, debug_all_path, scale=100)
    facade_shapely_list = []
    for f3d in facade_polygons_3d:
        f2d = project_3d(f3d)
        shp_f = to_shapely_polygon(f2d)
        shp_f = safe_buffer0(shp_f)
        if shp_f and not shp_f.is_empty:
            facade_shapely_list.append(shp_f)
    if not facade_shapely_list:
        print(f"  Invalid facade polygons after projection, skip {wall_id}")
        return wall_id
    try:
        facade_polygon = unary_union(facade_shapely_list)
        facade_polygon = safe_buffer0(facade_polygon)
    except Exception as e:
        print(f"  [Warning] union of facade polygons failed: {e}")
        return wall_id
    if not facade_polygon or facade_polygon.is_empty:
        print(f"  Facade polygon is empty after union, skip {wall_id}")
        return wall_id
    opening_shapely_list = []
    for op_3d in opening_polygons_3d:
        op_2d = project_3d(op_3d)
        shp_op = to_shapely_polygon(op_2d)
        shp_op = safe_buffer0(shp_op)
        if shp_op and not shp_op.is_empty:
            opening_shapely_list.append(shp_op)
    if not opening_shapely_list:
        print(f"  No valid opening polygons for {wall_id}, output facade only.")
        openings_union = None
    else:
        dist_threshold = 0.1
        grouped_openings = group_polygons_by_proximity(opening_shapely_list, dist_thresh=dist_threshold)
        if grouped_openings:
            debug_grouped_2d = []
            for g in grouped_openings:
                if g.geom_type == "Polygon":
                    debug_grouped_2d.append(np.array(g.exterior.coords))
                elif g.geom_type == "MultiPolygon":
                    for subg in g.geoms:
                        debug_grouped_2d.append(np.array(subg.exterior.coords))
            debug_grouped_path = os.path.join(debug_dir, f"debug_{wall_id}_grouped_openings.png")
            debug_save_multi_polygon_2d(debug_grouped_2d, debug_grouped_path, scale=100)
            try:
                openings_union = unary_union(grouped_openings)
                openings_union = safe_buffer0(openings_union)
            except Exception as e:
                print(f"  [Warning] union of grouped openings failed: {e}")
                openings_union = None
        else:
            print(f"  No grouped openings after proximity, skip {wall_id}")
            openings_union = None
    if openings_union and not openings_union.is_empty:
        try:
            facade_with_holes = facade_polygon.difference(openings_union)
            facade_with_holes = safe_buffer0(facade_with_holes)
        except Exception as e:
            print(f"  [Warning] difference failed: {e}")
            facade_with_holes = facade_polygon
    else:
        facade_with_holes = facade_polygon
    if not facade_with_holes or facade_with_holes.is_empty:
        print(f"  Facade with holes is empty, skip {wall_id}.")
        return wall_id
    if facade_with_holes.geom_type == "Polygon":
        polygons_to_export = [facade_with_holes]
    elif facade_with_holes.geom_type == "MultiPolygon":
        polygons_to_export = list(facade_with_holes.geoms)
    else:
        polygons_to_export = []
    all_rings_np = []
    for poly_obj in polygons_to_export:
        if poly_obj.exterior:
            all_rings_np.append(np.array(poly_obj.exterior.coords))
        for ring in poly_obj.interiors:
            all_rings_np.append(np.array(ring.coords))
    if not all_rings_np:
        print(f"  No exterior ring in final geometry, skip {wall_id}")
        return wall_id
    all_rings_np = np.vstack(all_rings_np)
    min_xy = np.min(all_rings_np, axis=0)
    max_xy = np.max(all_rings_np, axis=0)
    scale = 100
    width = int((max_xy[0] - min_xy[0]) * scale)
    height = int((max_xy[1] - min_xy[1]) * scale)
    if width < 1 or height < 1:
        print(f"  Invalid image size, skip {wall_id}")
        return wall_id
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    def ring_to_pixels(coords_2d):
        px = (coords_2d - min_xy) * scale
        px_flipped = np.column_stack((width - px[:, 0], height - px[:, 1]))
        return [tuple(p) for p in px_flipped]
    for poly_obj in polygons_to_export:
        if poly_obj.exterior:
            ext_coords = np.array(poly_obj.exterior.coords)
            ext_pixels = ring_to_pixels(ext_coords)
            draw.polygon(ext_pixels, fill=255)
        for ring in poly_obj.interiors:
            int_coords = np.array(ring.coords)
            ring_pixels = ring_to_pixels(int_coords)
            draw.polygon(ring_pixels, fill=0)
    out_path = os.path.join(output_dir, f"mask_{wall_id}.png")
    mask.save(out_path)
    print(f"  ==> Saved final mask for wall {wall_id} to {out_path}")
    return wall_id

def process_gml_file(gml_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug_masks")
    os.makedirs(debug_dir, exist_ok=True)
    tree = ET.parse(gml_path)
    root = tree.getroot()
    walls = root.findall(".//bldg:WallSurface", ns)
    xml_list = [ET.tostring(w) for w in walls]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [exe.submit(process_wall, xml, debug_dir, output_dir) for xml in xml_list]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {os.path.basename(gml_path)}"):
            pass
    print(f"Finished processing file: {gml_path}")

def main():
    input_dir = "/home/tang/code/CityGML2OBJv2/data/tum2twin-datasets/citygml/lod3-building-datasets"
    output_base_dir = "/home/tang/code/CityGML2OBJv2/mask_extraction_all"
    os.makedirs(output_base_dir, exist_ok=True)
    gml_files = glob.glob(os.path.join(input_dir, "*.gml"))
    for gml_path in gml_files:
        base_name = os.path.splitext(os.path.basename(gml_path))[0]
        output_dir = os.path.join(output_base_dir, base_name)
        print(f"\n=== Processing file: {gml_path} ===")
        process_gml_file(gml_path, output_dir)
        print(f"=== Finished processing file: {gml_path} ===\n")

if __name__ == "__main__":
    main()
