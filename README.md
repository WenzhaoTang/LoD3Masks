# CityGML Mask Extraction

A lightweight Python tool to extract 2D mask images of building façades (with window/door openings) from a CityGML file.

---

<table>
  <tr>
    <td align="center">
      <img src=" example_data/4906972/mask_DEBY_LOD2_4906972_c4547f98-389d-4617-a5ef-405fbc939e8f.png" alt="Image 1" width="200px"><br>
      4906972
    </td>
    <td align="center">
      <img src="/Users/Tang/code/LoD3Masks/ example_data/4906970/mask_DEBY_LOD2_4906970_c368b4d3-0164-4919-b93a-65c11c4df672.png" alt="Image 2" width="200px"><br>
      4906970
    </td>
    <td align="center">
      <img src=" example_data/4907507/mask_DEBY_LOD2_4907507_8ee10064-1c61-4081-948f-ca2915a1d26a.png" alt="Image 3" width="200px"><br>
      4907507
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src=" example_data/4907514/mask_DEBY_LOD2_4907514_9f0fe318-e654-43c1-873c-19793b0b3233.png" alt="Image 4" width="200px"><br>
      4907514
    </td>
    <td align="center">
      <img src=" example_data/4907516/mask_DEBY_LOD2_4907516_1f1f95ff-cd9e-44b4-8a72-ea9ed4d0b405.png" alt="Image 5" width="200px"><br>
      4907516
    </td>
    <td align="center">
      <img src=" example_data/4907520/mask_DEBY_LOD2_4907520_ea7be6c7-c269-489e-b51d-4153cb0c5e49.png" alt="Image 6" width="200px"><br>
      4907520
    </td>
  </tr>
</table>


## Core Method

1. **Parse GML**  
   Extract exterior and interior polygon coordinates for each `<bldg:WallSurface>`.

2. **PCA Projection**  
   Project 3D wall geometry into 2D via PCA.

3. **Shapely Processing**  
   Merge façade polygons into a single shape, group nearby openings (windows/doors), then subtract them to create holes.

4. **Rasterization**  
   Render the resulting polygon(s) as a binary mask image (white = façade, black = background).