# CityGML Mask Extraction

A lightweight Python tool to extract 2D mask images of building façades (with window/door openings) from a CityGML file.

---

## Core Method

1. **Parse GML**  
   Extract exterior and interior polygon coordinates for each `<bldg:WallSurface>`.

2. **PCA Projection**  
   Project 3D wall geometry into 2D via PCA.

3. **Shapely Processing**  
   Merge façade polygons into a single shape, group nearby openings (windows/doors), then subtract them to create holes.

4. **Rasterization**  
   Render the resulting polygon(s) as a binary mask image (white = façade, black = background).