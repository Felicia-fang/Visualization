import vtk
import plyfile
import json
import numpy as np

# Load the first JSON file with aggregation information
aggregation_file_path = "D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0000_00/scene0000_00.aggregation.json"
with open(aggregation_file_path, "r") as aggregation_file:
    aggregation_data = json.load(aggregation_file)

# Load the second JSON file with segmentation information
segmentation_file_path = "D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.0.010000.segs.json"
with open(segmentation_file_path, "r") as segmentation_file:
    segmentation_data = json.load(segmentation_file)

# Prompt user for a label
# user_label = input("Enter a label: ")

# # Find matching segments based on user input label
# matching_segments_dict = {}

# for seg_group in aggregation_data["segGroups"]:
#     if seg_group["label"] == user_label:
#         seg_id = seg_group["id"]
#         if seg_id not in matching_segments_dict:
#             matching_segments_dict[seg_id] = []
#         matching_segments_dict[seg_id].extend(seg_group["segments"])

# # Extract matching segments from the dictionary into a 2D list
# matching_segments = list(matching_segments_dict.values())

# print(matching_segments)

# matching_indices_dict = {}  # Dictionary to store matching indices for different seg_group["id"]
# segmentation_data = np.array(segmentation_data["segIndices"])
# seg_group_id = -1
# # Loop through each matching segment
# for seg_group in matching_segments:
#     seg_group_id = seg_group_id + 1
#     for matching_segment in seg_group:
#         positions = np.where(segmentation_data == matching_segment)[0]
#         if seg_group_id not in matching_indices_dict:
#             matching_indices_dict[seg_group_id] = []
#         matching_indices_dict[seg_group_id].extend(positions.tolist())
        
# Load the PLY file
# ply_file_path = "D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply"
ply_file_path = "./aligned_mesh2.ply"
with open(ply_file_path, "rb") as ply_file:
    plydata = plyfile.PlyData.read(ply_file)

# Extract vertex coordinates from PLY data
vertices = np.vstack([
    plydata["vertex"]["x"],
    plydata["vertex"]["y"],
    plydata["vertex"]["z"]
]).T

# Extract the coordinates of matching points
# matching_points = vertices[matching_indices]
# matching_points = vertices[matching_indices_dict[0]]



import vtk
import numpy as np

# Load the PLY file using vtkPLYReader
reader = vtk.vtkPLYReader()
# reader.SetFileName("D:/aaaaaaaaaaaaaaaaa/23-24/astar/1-2data/ScanNet/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply")
reader.SetFileName("./aligned_mesh1.ply")
reader.Update()

# Access label information from plydata
# labels = plydata["vertex"]["label"]

# Normalize RGB color values to the range [0, 1]
red = plydata["vertex"]["red"] / 255.0
green = plydata["vertex"]["green"] / 255.0
blue = plydata["vertex"]["blue"] / 255.0

# Create VTK points to store vertex coordinates
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

bounding_box_actor = vtk.vtkActor()
bounding_box_actor.SetMapper(bounding_box_mapper)
bounding_box_actor.GetProperty().SetColor(1, 1, 1)
bounding_box_actor.GetProperty().SetRepresentationToWireframe()

# Add mesh, bounding box, and point cloud actors to the renderer
renderer.AddActor(mesh_actor)
renderer.AddActor(bounding_box_actor)
renderer.AddActor(cloud_actor)
bbox=[[[3.22530055, 3.42303938, 1.11572492],
       [3.22530055, 1.58023506, 1.11572492],
       [2.6083262 , 1.58023506, 1.11572492],
       [2.6083262 , 3.42303938, 1.11572492],
       [3.22530055, 3.42303938, 0.82875049],
       [3.22530055, 1.58023506, 0.82875049],
       [2.6083262 , 1.58023506, 0.82875049],
       [2.6083262 , 3.42303938, 0.82875049]], [[-3.09104002,  1.91572744,  2.15139174],
       [-3.09104002, -1.94242972,  2.15139174],
       [-3.53185618, -1.94242972,  2.15139174],
       [-3.53185618,  1.91572744,  2.15139174],
       [-3.09104002,  1.91572744, -0.00895238],
       [-3.09104002, -1.94242972, -0.00895238],
       [-3.53185618, -1.94242972, -0.00895238],
       [-3.53185618,  1.91572744, -0.00895238]], [[-1.01790577, -3.37772441,  2.21514159],
       [-1.01790577, -3.59375834,  2.21514159],
       [-3.00905365, -3.59375834,  2.21514159],
       [-3.00905365, -3.37772441,  2.21514159],
       [-1.01790577, -3.37772441,  0.9384796 ],
       [-1.01790577, -3.59375834,  0.9384796 ],
       [-3.00905365, -3.59375834,  0.9384796 ],
       [-3.00905365, -3.37772441,  0.9384796 ]], [[-1.45443445, -2.79682958,  1.02040339],
       [-1.45443445, -3.48359072,  1.02040339],
       [-2.9345507 , -3.48359072,  1.02040339],
       [-2.9345507 , -2.79682958,  1.02040339],
       [-1.45443445, -2.79682958, -0.03816509],
       [-1.45443445, -3.48359072, -0.03816509],
       [-2.9345507 , -3.48359072, -0.03816509],
       [-2.9345507 , -2.79682958, -0.03816509]], [[-2.01706755,  3.15180409,  2.30798221],
       [-2.01706755,  1.65910518,  2.30798221],
       [-3.49348772,  1.65910518,  2.30798221],
       [-3.49348772,  3.15180409,  2.30798221],
       [-2.01706755,  3.15180409, -0.0299716 ],
       [-2.01706755,  1.65910518, -0.0299716 ],
       [-3.49348772,  1.65910518, -0.0299716 ],
       [-3.49348772,  3.15180409, -0.0299716 ]], [[ 3.29410076, -1.84635586,  0.88236982],
       [ 3.29410076, -2.52022511,  0.88236982],
       [ 2.84740496, -2.52022511,  0.88236982],
       [ 2.84740496, -1.84635586,  0.88236982],
       [ 3.29410076, -1.84635586, -0.02208006],
       [ 3.29410076, -2.52022511, -0.02208006],
       [ 2.84740496, -2.52022511, -0.02208006],
       [ 2.84740496, -1.84635586, -0.02208006]], [[ 2.91205871,  0.8180951 ,  0.38125998],
       [ 2.91205871,  0.45096754,  0.38125998],
       [ 2.5383054 ,  0.45096754,  0.38125998],
       [ 2.5383054 ,  0.8180951 ,  0.38125998],
       [ 2.91205871,  0.8180951 , -0.00400767],
       [ 2.91205871,  0.45096754, -0.00400767],
       [ 2.5383054 ,  0.45096754, -0.00400767],
       [ 2.5383054 ,  0.8180951 , -0.00400767]], [[3.22275603, 0.8128795 , 0.6564551 ],
       [3.22275603, 0.39762026, 0.6564551 ],
       [2.76513994, 0.39762026, 0.6564551 ],
       [2.76513994, 0.8128795 , 0.6564551 ],
       [3.22275603, 0.8128795 , 0.01012284],
       [3.22275603, 0.39762026, 0.01012284],
       [2.76513994, 0.39762026, 0.01012284],
       [2.76513994, 0.8128795 , 0.01012284]], [[-1.45000136, -2.73618162,  0.26406559],
       [-1.45000136, -3.23267138,  0.26406559],
       [-1.69538343, -3.23267138,  0.26406559],
       [-1.69538343, -2.73618162,  0.26406559],
       [-1.45000136, -2.73618162, -0.00823954],
       [-1.45000136, -3.23267138, -0.00823954],
       [-1.69538343, -3.23267138, -0.00823954],
       [-1.69538343, -2.73618162, -0.00823954]], [[3.26010633, 1.90405321, 1.81834996],
       [3.26010633, 0.67327881, 1.81834996],
       [2.32349229, 0.67327881, 1.81834996],
       [2.32349229, 1.90405321, 1.81834996],
       [3.26010633, 1.90405321, 0.00558281],
       [3.26010633, 0.67327881, 0.00558281],
       [2.32349229, 0.67327881, 0.00558281],
       [2.32349229, 1.90405321, 0.00558281]], [[ 0.47862649, -1.48009235,  0.81797326],
       [ 0.47862649, -3.47442359,  0.81797326],
       [-1.08157599, -3.47442359,  0.81797326],
       [-1.08157599, -1.48009235,  0.81797326],
       [ 0.47862649, -1.48009235, -0.01588041],
       [ 0.47862649, -3.47442359, -0.01588041],
       [-1.08157599, -3.47442359, -0.01588041],
       [-1.08157599, -1.48009235, -0.01588041]], [[ 3.30946112, -0.45964074,  2.36428571],
       [ 3.30946112, -1.50119221,  2.36428571],
       [ 2.91232157, -1.50119221,  2.36428571],
       [ 2.91232157, -0.45964074,  2.36428571],
       [ 3.30946112, -0.45964074, -0.00706005],
       [ 3.30946112, -1.50119221, -0.00706005],
       [ 2.91232157, -1.50119221, -0.00706005],
       [ 2.91232157, -0.45964074, -0.00706005]], [[-2.84952211, -2.2963233 ,  1.90920126],
       [-2.84952211, -3.51419687,  1.90920126],
       [-3.45077944, -3.51419687,  1.90920126],
       [-3.45077944, -2.2963233 ,  1.90920126],
       [-2.84952211, -2.2963233 , -0.02891207],
       [-2.84952211, -3.51419687, -0.02891207],
       [-3.45077944, -3.51419687, -0.02891207],
       [-3.45077944, -2.2963233 , -0.02891207]], [[3.26018989, 1.79103169, 2.38281119],
       [3.26018989, 0.81491438, 2.38281119],
       [2.59523308, 0.81491438, 2.38281119],
       [2.59523308, 1.79103169, 2.38281119],
       [3.26018989, 1.79103169, 1.72315824],
       [3.26018989, 0.81491438, 1.72315824],
       [2.59523308, 0.81491438, 1.72315824],
       [2.59523308, 1.79103169, 1.72315824]], [[ 1.11382118, -2.80553925,  2.31159639],
       [ 1.11382118, -3.51843059,  2.31159639],
       [ 0.75501809, -3.51843059,  2.31159639],
       [ 0.75501809, -2.80553925,  2.31159639],
       [ 1.11382118, -2.80553925,  1.41064572],
       [ 1.11382118, -3.51843059,  1.41064572],
       [ 0.75501809, -3.51843059,  1.41064572],
       [ 0.75501809, -2.80553925,  1.41064572]], [[2.70640659, 3.42373109, 0.93161553],
       [2.70640659, 1.57955527, 0.93161553],
       [2.59907007, 1.57955527, 0.93161553],
       [2.59907007, 3.42373109, 0.93161553],
       [2.70640659, 3.42373109, 0.044348  ],
       [2.70640659, 1.57955527, 0.044348  ],
       [2.59907007, 1.57955527, 0.044348  ],
       [2.59907007, 3.42373109, 0.044348  ]], [[3.08575296, 3.409024  , 2.40855527],
       [3.08575296, 1.71804929, 2.40855527],
       [2.81914043, 1.71804929, 2.40855527],
       [2.81914043, 3.409024  , 2.40855527],
       [3.08575296, 3.409024  , 1.35604954],
       [3.08575296, 1.71804929, 1.35604954],
       [2.81914043, 1.71804929, 1.35604954],
       [2.81914043, 3.409024  , 1.35604954]], [[ 3.40548897,  0.31716123,  1.91116476],
       [ 3.40548897, -0.36364052,  1.91116476],
       [ 3.30683851, -0.36364052,  1.91116476],
       [ 3.30683851,  0.31716123,  1.91116476],
       [ 3.40548897,  0.31716123, -0.00707483],
       [ 3.40548897, -0.36364052, -0.00707483],
       [ 3.30683851, -0.36364052, -0.00707483],
       [ 3.30683851,  0.31716123, -0.00707483]], [[ 2.6322822 , -1.4674015 ,  1.98277938],
       [ 2.6322822 , -1.6911149 ,  1.98277938],
       [ 1.61673075, -1.6911149 ,  1.98277938],
       [ 1.61673075, -1.4674015 ,  1.98277938],
       [ 2.6322822 , -1.4674015 , -0.01394463],
       [ 2.6322822 , -1.6911149 , -0.01394463],
       [ 1.61673075, -1.6911149 , -0.01394463],
       [ 1.61673075, -1.4674015 , -0.01394463]], [[ 3.39800000e+00,  3.42213362e-01,  1.96036065e+00],
       [ 3.39800000e+00, -4.34850067e-01,  1.96036065e+00],
       [ 3.20308661e+00, -4.34850067e-01,  1.96036065e+00],
       [ 3.20308661e+00,  3.42213362e-01,  1.96036065e+00],
       [ 3.39800000e+00,  3.42213362e-01,  2.38049030e-03],
       [ 3.39800000e+00, -4.34850067e-01,  2.38049030e-03],
       [ 3.20308661e+00, -4.34850067e-01,  2.38049030e-03],
       [ 3.20308661e+00,  3.42213362e-01,  2.38049030e-03]]]
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
    actor.GetProperty().SetColor(1, 0, 0)  # Set bounding box color to red

    renderer.AddActor(actor)
pre_bbox=[[[ 0.69252141, -1.43999955, -0.06432152],
       [ 0.69252141, -3.56589034, -0.06432152],
       [-1.26830187, -3.56589034, -0.06432152],
       [-1.26830187, -1.43999955, -0.06432152],
       [ 0.69252141, -1.43999955,  1.11987329],
       [ 0.69252141, -3.56589034,  1.11987329],
       [-1.26830187, -3.56589034,  1.11987329],
       [-1.26830187, -1.43999955,  1.11987329]], [[ 2.90729506,  3.00521707, -0.04159746],
       [ 2.90729506,  1.67158747, -0.04159746],
       [ 2.61774843,  1.67158747, -0.04159746],
       [ 2.61774843,  3.00521707, -0.04159746],
       [ 2.90729506,  3.00521707,  0.93110201],
       [ 2.90729506,  1.67158747,  0.93110201],
       [ 2.61774843,  1.67158747,  0.93110201],
       [ 2.61774843,  3.00521707,  0.93110201]], [[ 2.390434  , -1.44121861,  0.02670974],
       [ 2.390434  , -1.79337311,  0.02670974],
       [ 1.38448765, -1.79337311,  0.02670974],
       [ 1.38448765, -1.44121861,  0.02670974],
       [ 2.390434  , -1.44121861,  2.18654233],
       [ 2.390434  , -1.79337311,  2.18654233],
       [ 1.38448765, -1.79337311,  2.18654233],
       [ 1.38448765, -1.44121861,  2.18654233]], [[ 3.30164276, -1.84807376,  0.08165582],
       [ 3.30164276, -2.41505977,  0.08165582],
       [ 2.88013424, -2.41505977,  0.08165582],
       [ 2.88013424, -1.84807376,  0.08165582],
       [ 3.30164276, -1.84807376,  0.82784323],
       [ 3.30164276, -2.41505977,  0.82784323],
       [ 2.88013424, -2.41505977,  0.82784323],
       [ 2.88013424, -1.84807376,  0.82784323]], [[3.00281112, 1.7619497 , 0.07451284],
       [3.00281112, 0.87279685, 0.07451284],
       [2.3355058 , 0.87279685, 0.07451284],
       [2.3355058 , 1.7619497 , 0.07451284],
       [3.00281112, 1.7619497 , 1.72247231],
       [3.00281112, 0.87279685, 1.72247231],
       [2.3355058 , 0.87279685, 1.72247231],
       [2.3355058 , 1.7619497 , 1.72247231]], [[3.29460996, 2.98821729, 1.35423882],
       [3.29460996, 1.54662091, 1.35423882],
       [2.82029301, 1.54662091, 1.35423882],
       [2.82029301, 2.98821729, 1.35423882],
       [3.29460996, 2.98821729, 2.11467282],
       [3.29460996, 1.54662091, 2.11467282],
       [2.82029301, 1.54662091, 2.11467282],
       [2.82029301, 2.98821729, 2.11467282]], [[-1.84768281, -2.2351437 , -0.00821956],
       [-1.84768281, -2.94641014, -0.00821956],
       [-2.52103057, -2.94641014, -0.00821956],
       [-2.52103057, -2.2351437 , -0.00821956],
       [-1.84768281, -2.2351437 ,  0.86738773],
       [-1.84768281, -2.94641014,  0.86738773],
       [-2.52103057, -2.94641014,  0.86738773],
       [-2.52103057, -2.2351437 ,  0.86738773]], [[ 3.40686046,  0.47834302, -0.04452595],
       [ 3.40686046, -0.54270478, -0.04452595],
       [ 3.19525612, -0.54270478, -0.04452595],
       [ 3.19525612,  0.47834302, -0.04452595],
       [ 3.40686046,  0.47834302,  2.20031508],
       [ 3.40686046, -0.54270478,  2.20031508],
       [ 3.19525612, -0.54270478,  2.20031508],
       [ 3.19525612,  0.47834302,  2.20031508]], [[3.45179802, 3.75416373, 0.78630789],
       [3.45179802, 1.57206442, 0.78630789],
       [2.41372341, 1.57206442, 0.78630789],
       [2.41372341, 3.75416373, 0.78630789],
       [3.45179802, 3.75416373, 1.01085572],
       [3.45179802, 1.57206442, 1.01085572],
       [2.41372341, 1.57206442, 1.01085572],
       [2.41372341, 3.75416373, 1.01085572]]]
# Loop through each matching segment group, create bounding boxes, and add to renderer
for bbox_id in pre_bbox:

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
    actor.GetProperty().SetColor(0, 1, 0)  # Set bounding box color to red

    renderer.AddActor(actor)
# # Create text actor for labels and add to renderer
# label_text = vtk.vtkTextActor()
# label_text.GetTextProperty().SetFontSize(28)
# label_text.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # Set text color to white

# label_text.SetInput(user_label)  # Set text content

# # Set text alignment properties
# label_text.GetTextProperty().SetVerticalJustificationToBottom()
# label_text.GetTextProperty().SetJustificationToLeft()

# # Calculate label text position and adjust Z-axis to avoid occlusion
# label_text_position = (min_coords + max_coords) / 2 
# label_text_position[2] = max_coords[2] 
# label_text_position[1] -= 0.05 * (max_coords[1] - min_coords[1]) 

# label_text.SetPosition(label_text_position[0], label_text_position[1])
# renderer.AddActor2D(label_text)

# Create a VTK render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create a VTK render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Render the scene
render_window.Render()

# Start interaction
render_window_interactor.Start()
