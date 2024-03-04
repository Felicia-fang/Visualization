import vtk
import plyfile
import json
import numpy as np

# Load the first JSON file with aggregation information
aggregation_file_path = "D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0062_00/scene0062_00.aggregation.json"
with open(aggregation_file_path, "r") as aggregation_file:
    aggregation_data = json.load(aggregation_file)

# Load the second JSON file with segmentation information
segmentation_file_path = "D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0062_00/scene0062_00_vh_clean_2.0.010000.segs.json"
with open(segmentation_file_path, "r") as segmentation_file:
    segmentation_data = json.load(segmentation_file)
ply_file_path = "./scene0062_00_aligned_mesh2.ply"
with open(ply_file_path, "rb") as ply_file:
    plydata = plyfile.PlyData.read(ply_file)

# Extract vertex coordinates from PLY data
vertices = np.vstack([
    plydata["vertex"]["x"],
    plydata["vertex"]["y"],
    plydata["vertex"]["z"]
]).T

# Load the PLY file using vtkPLYReader
reader = vtk.vtkPLYReader()
# reader.SetFileName("D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply")
reader.SetFileName("./scene0062_00_aligned_mesh2.ply")
# reader.Update()


red = plydata["vertex"]["red"] / 255.0
green = plydata["vertex"]["green"] / 255.0
blue = plydata["vertex"]["blue"] / 255.0


vtk_points = vtk.vtkPoints()
for vertex in vertices:
    vtk_points.InsertNextPoint(vertex)

# Create VTK cell array for the point cloud
cloud_verts = vtk.vtkCellArray()
for i in range(len(vertices)):
    cloud_verts.InsertNextCell(1)
    cloud_verts.InsertCellPoint(i)

# Create VTK colors array for RGB vertex colors
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")

# Populate colors array with RGB tuples
for r, g, b in zip(red, green, blue):
    colors.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))

# Create VTK PolyData for the point cloud
point_cloud = vtk.vtkPolyData()
point_cloud.SetPoints(vtk_points)
point_cloud.SetVerts(cloud_verts)
point_cloud.GetPointData().SetScalars(colors)

# Create VTK PolyDataMapper for the point cloud
cloud_mapper = vtk.vtkPolyDataMapper()
cloud_mapper.SetInputData(point_cloud)

# Create VTK Actor for the point cloud
cloud_actor = vtk.vtkActor()
cloud_actor.SetMapper(cloud_mapper)

# Create a VTK renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(1.0, 1.0, 1.0)
# Create a bounding box for the mesh
bounds = reader.GetOutput().GetBounds()
bounding_box = vtk.vtkCubeSource()
bounding_box.SetBounds(bounds)

# Create a PolyDataMapper and Actor for the mesh
mesh_mapper = vtk.vtkPolyDataMapper()
mesh_mapper.SetInputConnection(reader.GetOutputPort())

mesh_actor = vtk.vtkActor()
mesh_actor.SetMapper(mesh_mapper)

# Create a PolyDataMapper and Actor for the bounding box
bounding_box_mapper = vtk.vtkPolyDataMapper()
bounding_box_mapper.SetInputConnection(bounding_box.GetOutputPort())

renderer.AddActor(mesh_actor)
# renderer.AddActor(bounding_box_actor)
renderer.AddActor(cloud_actor)


def get_box_corners(center, dimensions):
    x, y, z, width, height, depth = center[0], center[1], center[2], dimensions[0], dimensions[1], dimensions[2]
    corners = np.array([
        [x - width / 2, y - height / 2, z - depth / 2],
        [x + width / 2, y - height / 2, z - depth / 2],
        [x - width / 2, y + height / 2, z - depth / 2],
        [x + width / 2, y + height / 2, z - depth / 2],
        [x - width / 2, y - height / 2, z + depth / 2],
        [x + width / 2, y - height / 2, z + depth / 2],
        [x - width / 2, y + height / 2, z + depth / 2],
        [x + width / 2, y + height / 2, z + depth / 2]
    ])
    return corners


gt_bboxes = np.array([[[0.76312193,  1.15269831,  1.07095996],
                       [0.76312193,  0.56106469,  1.07095996],
                       [-1.13912836,  0.56106469,  1.07095996],
                       [-1.13912836,  1.15269831,  1.07095996],
                       [0.76312193,  1.15269831,  0.67129359],
                       [0.76312193,  0.56106469,  0.67129359],
                       [-1.13912836,  0.56106469,  0.67129359],
                       [-1.13912836,  1.15269831,  0.67129359]], [[-0.22681688,  0.83066726,  0.66352397],
                                                                  [-0.22681688,  0.62376714,
                                                                   0.66352397],
                                                                  [-0.57924761,  0.62376714,
                                                                   0.66352397],
                                                                  [-0.57924761,  0.83066726,
                                                                   0.66352397],
                                                                  [-0.22681688,  0.83066726,
                                                                   0.00735432],
                                                                  [-0.22681688,  0.62376714,
                                                                   0.00735432],
                                                                  [-0.57924761,  0.62376714,
                                                                   0.00735432],
                                                                  [-0.57924761,  0.83066726,  0.00735432]], [[-0.01585019, -1.31432015,  0.54387677],
                                                                                                             [-0.01585019, -1.46136421,
                                                                                                              0.54387677],
                                                                                                             [-0.35243565, -1.46136421,
                                                                                                              0.54387677],
                                                                                                             [-0.35243565, -1.31432015,
                                                                                                              0.54387677],
                                                                                                             [-0.01585019, -1.31432015,
                                                                                                              0.21096718],
                                                                                                             [-0.01585019, -1.46136421,
                                                                                                              0.21096718],
                                                                                                             [-0.35243565, -1.46136421,
                                                                                                              0.21096718],
                                                                                                             [-0.35243565, -1.31432015,  0.21096718]], [[-0.46645615, -0.66869915,  0.86686635],
                                                                                                                                                        [-0.46645615, -1.44348061,
                                                                                                                                                         0.86686635],
                                                                                                                                                        [-0.94790891, -1.44348061,
                                                                                                                                                         0.86686635],
                                                                                                                                                        [-0.94790891, -0.66869915,
                                                                                                                                                         0.86686635],
                                                                                                                                                        [-0.46645615, -0.66869915,
                                                                                                                                                         0.00518626],
                                                                                                                                                        [-0.46645615, -1.44348061,
                                                                                                                                                         0.00518626],
                                                                                                                                                        [-0.94790891, -1.44348061,
                                                                                                                                                         0.00518626],
                                                                                                                                                        [-0.94790891, -0.66869915,  0.00518626]], [[9.45425749e-01,  3.14522550e-01,  2.06342244e+00],
                                                                                                                                                                                                   [9.45425749e-01, -6.05037287e-01,
                                                                                                                                                                                                    2.06342244e+00],
                                                                                                                                                                                                   [8.47442746e-01, -6.05037287e-01,
                                                                                                                                                                                                    2.06342244e+00],
                                                                                                                                                                                                   [8.47442746e-01,  3.14522550e-01,
                                                                                                                                                                                                    2.06342244e+00],
                                                                                                                                                                                                   [9.45425749e-01,
                                                                                                                                                                                                    3.14522550e-01, -7.24792480e-04],
                                                                                                                                                                                                   [9.45425749e-01, -
                                                                                                                                                                                                    6.05037287e-01, -7.24792480e-04],
                                                                                                                                                                                                   [8.47442746e-01, -
                                                                                                                                                                                                    6.05037287e-01, -7.24792480e-04],
                                                                                                                                                                                                   [8.47442746e-01,  3.14522550e-01, -7.24792480e-04]], [[0.50497438, 1.07363531, 0.90460482],
                                                                                                                                                                                                                                                         [0.50497438, 0.75249723,
                                                                                                                                                                                                                                                          0.90460482],
                                                                                                                                                                                                                                                         [0.10061704, 0.75249723,
                                                                                                                                                                                                                                                          0.90460482],
                                                                                                                                                                                                                                                         [0.10061704, 1.07363531,
                                                                                                                                                                                                                                                          0.90460482],
                                                                                                                                                                                                                                                         [0.50497438, 1.07363531,
                                                                                                                                                                                                                                                          0.67566285],
                                                                                                                                                                                                                                                         [0.50497438, 0.75249723,
                                                                                                                                                                                                                                                          0.67566285],
                                                                                                                                                                                                                                                         [0.10061704, 0.75249723,
                                                                                                                                                                                                                                                          0.67566285],
                                                                                                                                                                                                                                                         [0.10061704, 1.07363531, 0.67566285]], [[0.94490337,  0.33618392,  2.1046834],
                                                                                                                                                                                                                                                                                                 [0.94490337, -0.65291058,  2.1046834],
                                                                                                                                                                                                                                                                                                 [0.7285794, -0.65291058,  2.1046834],
                                                                                                                                                                                                                                                                                                 [0.7285794,  0.33618392,  2.1046834],
                                                                                                                                                                                                                                                                                                 [0.94490337,  0.33618392,  0.00675344],
                                                                                                                                                                                                                                                                                                 [0.94490337, -0.65291058,  0.00675344],
                                                                                                                                                                                                                                                                                                 [0.7285794, -0.65291058,  0.00675344],
                                                                                                                                                                                                                                                                                                 [0.7285794,  0.33618392,  0.00675344]]]

                     )
sdc_bbox = np.array([[[0.97401421,  0.31821075, -0.00338639],
                      [0.97401421, -0.63941523, -0.00338639],
                      [0.77918489, -0.63941523, -0.00338639],
                      [0.77918489,  0.31821075, -0.00338639],
                      [0.97401421,  0.31821075,  2.07481993],
                      [0.97401421, -0.63941523,  2.07481993],
                      [0.77918489, -0.63941523,  2.07481993],
                      [0.77918489,  0.31821075,  2.07481993]], [[0.11010943, -1.23678163,  0.23647104],
                                                                [0.11010943, -1.45325705,
                                                                 0.23647104],
                                                                [-0.256362, -1.45325705,
                                                                 0.23647104],
                                                                [-0.256362, -1.23678163,
                                                                 0.23647104],
                                                                [0.11010943, -1.23678163,
                                                                    0.68834629],
                                                                [0.11010943, -1.45325705,
                                                                    0.68834629],
                                                                [-0.256362, -1.45325705,
                                                                 0.68834629],
                                                                [-0.256362, -1.23678163,  0.68834629]], [[-0.25412004,  0.90039312,  0.04411824],
                                                                                                         [-0.25412004,  0.52694405,
                                                                                                          0.04411824],
                                                                                                         [-0.64520533,  0.52694405,
                                                                                                          0.04411824],
                                                                                                         [-0.64520533,  0.90039312,
                                                                                                          0.04411824],
                                                                                                         [-0.25412004,  0.90039312,
                                                                                                          0.57320826],
                                                                                                         [-0.25412004,  0.52694405,
                                                                                                          0.57320826],
                                                                                                         [-0.64520533,  0.52694405,
                                                                                                          0.57320826],
                                                                                                         [-0.64520533,  0.90039312,  0.57320826]], [[0.34106284,  0.99804856,  0.66451078],
                                                                                                                                                    [0.34106284,  0.75172196,
                                                                                                                                                     0.66451078],
                                                                                                                                                    [-0.17764243,  0.75172196,
                                                                                                                                                     0.66451078],
                                                                                                                                                    [-0.17764243,  0.99804856,
                                                                                                                                                     0.66451078],
                                                                                                                                                    [0.34106284,  0.99804856,  0.95265168],
                                                                                                                                                    [0.34106284,  0.75172196,  0.95265168],
                                                                                                                                                    [-0.17764243,  0.75172196,
                                                                                                                                                     0.95265168],
                                                                                                                                                    [-0.17764243,  0.99804856,  0.95265168]], [[-0.41906091, -0.64117873,  0.02725529],
                                                                                                                                                                                               [-0.41906091, -1.45986307,
                                                                                                                                                                                                0.02725529],
                                                                                                                                                                                               [-0.97607318, -1.45986307,
                                                                                                                                                                                                0.02725529],
                                                                                                                                                                                               [-0.97607318, -0.64117873,
                                                                                                                                                                                                0.02725529],
                                                                                                                                                                                               [-0.41906091, -0.64117873,
                                                                                                                                                                                                0.93540997],
                                                                                                                                                                                               [-0.41906091, -1.45986307,
                                                                                                                                                                                                0.93540997],
                                                                                                                                                                                               [-0.97607318, -1.45986307,
                                                                                                                                                                                                0.93540997],
                                                                                                                                                                                               [-0.97607318, -0.64117873,  0.93540997]]]

                    )
new_bboxes = np.array([[-0.70506406, -1.06950533, 0.45482075, 0.45097864, 0.78539933, 0.85680487],
                       #    [0.04709997, 0.83833051, 0.7845158,
                       #     0.45325837, 0.37434103, 0.22402154],
                       [0.89966363, -0.17883793, 1.10911059,
                        0.27108923, 1.13971701, 2.16547714],
                       [0.21965055, 0.84110165, 0.69340396,
                           0.61927754, 0.35294079, 0.23437782],
                       [-0.22143084, -1.3969543, 0.39106372,
                        0.32679064, 0.13097521, 0.34096281],
                       #    [-0.4707121, 0.6951133, 0.36803377,
                       #     0.30662095, 0.16116183, 0.62471881],
                       [-0.169596, 0.86847478, 0.86569542,
                        1.83218121, 0.49761277, 0.37470557],
                       #    [-0.02315431, 0.95496577, 0.58109951,
                       #     0.40691967, 0.33262408, 0.25104806],
                       #    [0.13032942, 0.73824608, 0.60429507,
                       #     0.39611349, 0.32696838, 0.25066718],
                       #    [0.41888, 0.85420185, 0.7002424,
                       #        0.43941194, 0.35727782, 0.25142636],
                       [-0.3452062, 0.75628871, 0.36810866,
                           0.35767574, 0.17448502, 0.63078877]
                       ])

# Loop through each matching segment group, create bounding boxes, and add to renderer
for bbox_id in gt_bboxes:

    # Define the corners of the bounding box
    corners = bbox_id

    # Create VTK Points for bounding box corners
    vtk_corners = vtk.vtkPoints()
    for corner in corners:
        vtk_corners.InsertNextPoint(corner)

    # Define line indices for bounding box edges
    line_indices = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    # Create VTK CellArray for bounding box lines
    lines = vtk.vtkCellArray()
    for indices in line_indices:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(indices[0])
        lines.InsertCellPoint(indices[1])

    # Create VTK PolyData for bounding box lines
    bounding_box_polydata = vtk.vtkPolyData()
    bounding_box_polydata.SetPoints(vtk_corners)
    bounding_box_polydata.SetLines(lines)

    # Create a VTK PolyDataMapper and Actor for the bounding box
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(bounding_box_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0.3, 0.3)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(5)
    # renderer.AddActor(actor)

for bbox_id in sdc_bbox:

    # Define the corners of the bounding box
    corners = bbox_id

    # Create VTK Points for bounding box corners
    vtk_corners = vtk.vtkPoints()
    for corner in corners:
        vtk_corners.InsertNextPoint(corner)

    # Define line indices for bounding box edges
    line_indices = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    # Create VTK CellArray for bounding box lines
    lines = vtk.vtkCellArray()
    for indices in line_indices:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(indices[0])
        lines.InsertCellPoint(indices[1])

    # Create VTK PolyData for bounding box lines
    bounding_box_polydata = vtk.vtkPolyData()
    bounding_box_polydata.SetPoints(vtk_corners)
    bounding_box_polydata.SetLines(lines)

    # Create a VTK PolyDataMapper and Actor for the bounding box
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(bounding_box_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 1)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(4)
    # renderer.AddActor(actor)

    corners = []
    # Calculate and print corners for each bounding box
    for bbox in new_bboxes:
        center = bbox[:3]
        dimensions = bbox[3:6]
        corner = get_box_corners(center, dimensions)
        corners.append(corner)
    bbox = corners

# Loop through each matching segment group, create bounding boxes, and add to renderer
for bbox_id in bbox:

    # Define the corners of the bounding box
    corners = bbox_id

    # Create VTK Points for bounding box corners
    vtk_corners = vtk.vtkPoints()
    for corner in corners:
        vtk_corners.InsertNextPoint(corner)

    # Define line indices for bounding box edges
    line_indices = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom edges
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    # Create VTK CellArray for bounding box lines
    lines = vtk.vtkCellArray()
    for indices in line_indices:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(indices[0])
        lines.InsertCellPoint(indices[1])

    # Create VTK PolyData for bounding box lines
    bounding_box_polydata = vtk.vtkPolyData()
    bounding_box_polydata.SetPoints(vtk_corners)
    bounding_box_polydata.SetLines(lines)

    # Create a VTK PolyDataMapper and Actor for the bounding box
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(bounding_box_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 1)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(4)
    # renderer.AddActor(actor)


def get_rectangle_corners(rectangle, thickness=0.1):
    center = rectangle[:3]
    direction = rectangle[3:6]
    width = rectangle[6]
    height = rectangle[7]

    # Calculate the direction of the width and height
    width_direction = np.array([-direction[1], direction[0], 0])
    height_direction = np.cross(direction, width_direction)

    # Normalize the directions
    width_direction /= np.linalg.norm(width_direction)
    height_direction /= np.linalg.norm(height_direction)

    # Calculate the corners
    half_width = width / 2
    half_height = height / 2
    half_thickness = thickness / 2
    corners = [
        center - half_width * width_direction - half_height *
        height_direction - half_thickness * direction,
        center + half_width * width_direction - half_height *
        height_direction - half_thickness * direction,
        center + half_width * width_direction + half_height *
        height_direction - half_thickness * direction,
        center - half_width * width_direction + half_height *
        height_direction - half_thickness * direction,
        center - half_width * width_direction - half_height *
        height_direction + half_thickness * direction,
        center + half_width * width_direction - half_height *
        height_direction + half_thickness * direction,
        center + half_width * width_direction + half_height *
        height_direction + half_thickness * direction,
        center - half_width * width_direction + half_height *
        height_direction + half_thickness * direction,
    ]

    return corners


# Your rectangles垂直
rectangles = np.array([[7.29177345e-01, -1.61363398e-01, 1.08127650e+00, -9.99998348e-01,
                        -1.81765324e-03, 0.00000000e+00, 2.49519955e+00, 2.16711869e+00],
                       [-1.93753450e-01, -1.41163423e+00,  1.08157050e+00, -2.88918973e-03,
                        9.99995826e-01,  0.00000000e+00, 1.85040444e+00, 2.16711869e+00],
                       [-1.95205553e-01,  1.08668286e+00, 1.08066950e+00, -4.86643422e-04,
                        -9.99999882e-01,  0.00000000e+00, 1.84423094e+00,  2.16711869e+00],
                       [-1.11813635e+00, -1.63587967e-01, 1.08096350e+00, 9.99999788e-01,
                        -6.51860738e-04, 0.00000000e+00, 2.50143961e+00, 2.16711869e+00]
                       ])
corners = []
# Calculate and print corners for each rectangle
for rectangle in rectangles:
    corner = get_rectangle_corners(rectangle)
    corners.append(corner)
bbox = corners
for bbox_id in bbox:

    # Define the corners of the bounding box
    corners = bbox_id

    # Create VTK Points for bounding box corners
    vtk_corners = vtk.vtkPoints()
    for corner in corners:
        vtk_corners.InsertNextPoint(corner)

    # Define line indices for bounding box edges
    line_indices = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    # Create VTK CellArray for bounding box lines
    lines = vtk.vtkCellArray()
    for indices in line_indices:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(indices[0])
        lines.InsertCellPoint(indices[1])

    # Create VTK PolyData for bounding box lines
    bounding_box_polydata = vtk.vtkPolyData()
    bounding_box_polydata.SetPoints(vtk_corners)
    bounding_box_polydata.SetLines(lines)

    # Create a VTK PolyDataMapper and Actor for the bounding box
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(bounding_box_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(3)
    renderer.AddActor(actor)


horizontal_quads = [[[-1.11768830e+00, 1.08791279e+00, 2.16407200e+00],
                     [7.26542398e-01, 1.08701553e+00, 2.16438500e+00],
                     [7.31076868e-01, -1.40817942e+00, 2.16528600e+00],
                     [-1.11931919e+00, -1.41352613e+00, 2.16497300e+00],
                     [-1.11768830e+00, 1.08791279e+00, 2.26407200e+00],
                     [7.26542398e-01, 1.08701553e+00, 2.26438500e+00],
                     [7.31076868e-01, -1.40817942e+00, 2.26528600e+00],
                     [-1.11931919e+00, -1.41352613e+00, 2.26497300e+00]],

                    [[-1.11695350e+00, 1.08635019e+00, -3.04600000e-03],
                     [7.27277190e-01, 1.08545294e+00, -2.73300000e-03],
                     [7.31812925e-01, -1.40974264e+00, -1.83200000e-03],
                     [-1.11858440e+00, -1.41508872e+00, -2.14500000e-03],
                     [-1.11695350e+00, 1.08635019e+00, -103.04600000e-03],
                     [7.27277190e-01, 1.08545294e+00, -102.73300000e-03],
                     [7.31812925e-01, -1.40974264e+00, -101.83200000e-03],
                     [-1.11858440e+00, -1.41508872e+00, -102.14500000e-03]]]


for bbox_id in horizontal_quads:

    # Define the corners of the bounding box
    corners = bbox_id

    # Create VTK Points for bounding box corners
    vtk_corners = vtk.vtkPoints()
    for corner in corners:
        vtk_corners.InsertNextPoint(corner)

    # Define line indices for bounding box edges
    line_indices = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]

    # Create VTK CellArray for bounding box lines
    lines = vtk.vtkCellArray()
    for indices in line_indices:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(indices[0])
        lines.InsertCellPoint(indices[1])

    # Create VTK PolyData for bounding box lines
    bounding_box_polydata = vtk.vtkPolyData()
    bounding_box_polydata.SetPoints(vtk_corners)
    bounding_box_polydata.SetLines(lines)

    # Create a VTK PolyDataMapper and Actor for the bounding box
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(bounding_box_polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(3)
    renderer.AddActor(actor)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create a VTK render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(style)

# Render the scene
render_window.Render()

# Start interaction
render_window_interactor.Start()
