import os
import sys
import glob
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from shapely.ops import unary_union
from shapely.errors import GEOSException
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from utils import (
    parse_poslist,
    extract_polygon_exterior_3d,
    extract_polygon_interior_3d,
    extract_all_polygons_recursively,
    to_shapely_polygon,
    safe_buffer0,
    safe_intersects,
    safe_distance,
    debug_save_multi_polygon_2d,
    compute_polygon_normal,
    define_axes_from_normal,
    project_points,
    ns,
    SHIFT,
    DEBUG,
    MASK_TYPE
)

def process_wall(xml_bytes, out_dir, debug_dir):
    wall = ET.fromstring(xml_bytes)
    wall_id = wall.attrib.get("{http://www.opengis.net/gml}id", "unknown")
    
    facade_polys_3d = extract_all_polygons_recursively(wall, ns)
    if not facade_polys_3d:
        return wall_id

    opening_polys_3d = []
    for poly in wall.findall(".//gml:Polygon", ns):
        ints = extract_polygon_interior_3d(poly)
        if ints:
            opening_polys_3d.extend(ints)
    for op in wall.findall(".//bldg:opening", ns):
        opening_polys_3d.extend(extract_all_polygons_recursively(op, ns))
    
    door_polys_3d = []
    for door in wall.findall(".//bldg:Door", ns):
        d = extract_all_polygons_recursively(door, ns)
        door_polys_3d.extend(d)
        opening_polys_3d.extend(d)
    window_polys_3d = []
    for window in wall.findall(".//bldg:Window", ns):
        w = extract_all_polygons_recursively(window, ns)
        window_polys_3d.extend(w)
        opening_polys_3d.extend(w)
    
    if not door_polys_3d and not window_polys_3d:
        return wall_id

    normal = compute_polygon_normal(facade_polys_3d[0])
    if normal is None:
        return wall_id
    axes = define_axes_from_normal(normal)
    if axes is None:
        return wall_id
    x_axis, y_axis = axes

    def project_3d(pts):
        return project_points(pts, x_axis, y_axis)

    # Save debug image if enabled
    debug_facade_2d = [project_3d(pts) for pts in facade_polys_3d]
    debug_opening_2d = [project_3d(pts) for pts in opening_polys_3d]
    if DEBUG:
        debug_all_path = os.path.join(debug_dir, f"debug_{wall_id}_all_polygons.png")
        debug_save_multi_polygon_2d(debug_facade_2d + debug_opening_2d, debug_all_path, scale=100)

    facade_shapes = []
    for pts in facade_polys_3d:
        pts2d = project_3d(pts)
        shp = to_shapely_polygon(pts2d)
        shp = safe_buffer0(shp)
        if shp is not None and not shp.is_empty:
            facade_shapes.append(shp)
    if not facade_shapes:
        return wall_id
    try:
        facade_union = unary_union(facade_shapes)
        facade_union = safe_buffer0(facade_union)
    except GEOSException:
        return wall_id
    if facade_union is None or facade_union.is_empty:
        return wall_id

    min_xy_building = np.array([facade_union.bounds[0], facade_union.bounds[1]])
    max_xy_building = np.array([facade_union.bounds[2], facade_union.bounds[3]])
    scale_factor = 100
    width_building = int((max_xy_building[0] - min_xy_building[0]) * scale_factor)
    height_building = int((max_xy_building[1] - min_xy_building[1]) * scale_factor)
    if width_building < 1 or height_building < 1:
        return wall_id

    def ring_to_pixels_building(coords):
        pts = (np.array(coords) - min_xy_building) * scale_factor
        pts_flipped = np.column_stack((width_building - pts[:, 0], height_building - pts[:, 1]))
        return [tuple(p) for p in pts_flipped]

    opening_shapes = []
    for pts in opening_polys_3d:
        pts2d = project_3d(pts)
        shp = to_shapely_polygon(pts2d)
        shp = safe_buffer0(shp)
        if shp is not None and not shp.is_empty:
            opening_shapes.append(shp)
    if opening_shapes:
        try:
            openings_union = unary_union(opening_shapes)
            openings_union = safe_buffer0(openings_union)
        except GEOSException:
            openings_union = None
    else:
        openings_union = None

    if (MASK_TYPE in ["all", "full"]) and (door_polys_3d or window_polys_3d):
        full_mask_img = Image.new("RGB", (width_building, height_building), (0, 0, 0))
        draw_full = ImageDraw.Draw(full_mask_img)
        for pts in door_polys_3d:
            pts2d = project_3d(pts)
            pixel_pts = ring_to_pixels_building(pts2d)
            draw_full.polygon(pixel_pts, fill=(255, 0, 0))
        for pts in window_polys_3d:
            pts2d = project_3d(pts)
            pixel_pts = ring_to_pixels_building(pts2d)
            draw_full.polygon(pixel_pts, fill=(0, 0, 255))
        full_out_path = os.path.join(out_dir, f"mask_{wall_id}_full.png")
        full_mask_img.save(full_out_path)

    if (MASK_TYPE in ["all", "door"]) and door_polys_3d:
        door_shapes = []
        for pts in door_polys_3d:
            pts2d = project_3d(pts)
            shp = to_shapely_polygon(pts2d)
            shp = safe_buffer0(shp)
            if shp is not None and not shp.is_empty:
                door_shapes.append(shp)
        if door_shapes:
            door_union = unary_union(door_shapes)
            door_union = safe_buffer0(door_union)
            if door_union is not None and not door_union.is_empty:
                if door_union.geom_type == "Polygon":
                    door_polys = [door_union]
                elif door_union.geom_type == "MultiPolygon":
                    door_polys = list(door_union.geoms)
                else:
                    door_polys = []
                door_mask_img = Image.new("L", (width_building, height_building), 0)
                door_draw = ImageDraw.Draw(door_mask_img)
                for poly in door_polys:
                    if poly.exterior is not None:
                        door_draw.polygon(ring_to_pixels_building(poly.exterior.coords), fill=255)
                    for ring in poly.interiors:
                        door_draw.polygon(ring_to_pixels_building(ring.coords), fill=0)
                door_out_path = os.path.join(out_dir, f"mask_{wall_id}_door.png")
                door_mask_img.save(door_out_path)
    
    if (MASK_TYPE in ["all", "window"]) and window_polys_3d:
        window_shapes = []
        for pts in window_polys_3d:
            pts2d = project_3d(pts)
            shp = to_shapely_polygon(pts2d)
            shp = safe_buffer0(shp)
            if shp is not None and not shp.is_empty:
                window_shapes.append(shp)
        if window_shapes:
            window_union = unary_union(window_shapes)
            window_union = safe_buffer0(window_union)
            if window_union is not None and not window_union.is_empty:
                if window_union.geom_type == "Polygon":
                    window_polys = [window_union]
                elif window_union.geom_type == "MultiPolygon":
                    window_polys = list(window_union.geoms)
                else:
                    window_polys = []
                window_mask_img = Image.new("L", (width_building, height_building), 0)
                window_draw = ImageDraw.Draw(window_mask_img)
                for poly in window_polys:
                    if poly.exterior is not None:
                        window_draw.polygon(ring_to_pixels_building(poly.exterior.coords), fill=255)
                    for ring in poly.interiors:
                        window_draw.polygon(ring_to_pixels_building(ring.coords), fill=0)
                window_out_path = os.path.join(out_dir, f"mask_{wall_id}_window.png")
                window_mask_img.save(window_out_path)
    
    return wall_id

def process_building(building, building_out_dir, building_debug_dir):
    building_id = building.attrib.get("{http://www.opengis.net/gml}id", "unknown_building")
    os.makedirs(building_out_dir, exist_ok=True)
    wall_elements = building.findall(".//bldg:WallSurface", ns)
    if not wall_elements:
        return
    ctx = get_context('fork')
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as exe:
        futures = [exe.submit(process_wall, ET.tostring(w), building_out_dir, building_out_dir)
                   for w in wall_elements]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

def process_gml_file(gml_path, output_base_dir):
    tree = ET.parse(gml_path)
    root = tree.getroot()
    buildings = root.findall(".//bldg:Building", ns)
    for building in buildings:
        building_id = building.attrib.get("{http://www.opengis.net/gml}id", "unknown_building")
        building_out_dir = os.path.join(output_base_dir, building_id)
        process_building(building, building_out_dir, building_out_dir)

def main():
    parser = argparse.ArgumentParser(description="CityGML Mask Extraction Tool")
    parser.add_argument("--input", required=True,
                        help="Path to input CityGML file or directory containing .gml files")
    parser.add_argument("--output", required=True,
                        help="Output directory for mask images")
    args = parser.parse_args()

    input_path = args.input
    output_base_dir = args.output
    os.makedirs(output_base_dir, exist_ok=True)

    if os.path.isdir(input_path):
        gml_files = glob.glob(os.path.join(input_path, "*.gml"))
    else:
        gml_files = [input_path]

    for gml_path in gml_files:
        process_gml_file(gml_path, output_base_dir)

if __name__ == "__main__":
    main()
