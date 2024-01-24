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
ply_file_path = "./aligned_mesh2.ply"
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
reader.SetFileName("./aligned_mesh1.ply")
reader.Update()


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


pseudo_bboxes = np.array([[3.06977463, -1.00425863, 1.16271257, 0.33900686, 1.03994321, 2.36064187],
                          [-0.26709002, -2.46302438, 0.38261265,
                              1.62359235,  2.06804298, 0.81759844],
                          [-2.12048197, -3.47512865, 1.60746253,
                              1.86930224, 0.26017974, 1.21202522],
                          [3.33823085, -0.03114552, 0.98311472,
                              0.16085008, 0.71703392, 1.91622867],
                          [2.97887707,  2.58704782,  1.88537848,
                              0.28915297,  1.69949674,  0.99758442],
                          [-3.1261375, -2.93652248, 0.98343062,
                              0.61148772, 1.4350847, 1.85989604],
                          [2.14187074, -1.5898056,  0.94584292,
                              0.99050904, 0.26446017, 1.99149373],
                          [-2.24088407, -3.12897086, 0.49564067,
                              1.46795521, 0.67877429, 1.03588973],
                          [-2.75840712, 2.34486651, 1.16092169,
                              1.49659561, 1.64514756,  2.37845567],
                          [2.85172367, 1.29112542,  2.05511475,
                              0.72190918, 1.07372802, 0.65205318],
                          [2.64781022, 2.51344061, 0.49312156,
                              0.15103995, 1.85338032,  0.85791148],
                          [-3.32199645, -0.05123363, 1.08958352,
                              0.62279129, 3.97767286,  2.15252679],
                          [0.9336133, -3.12980509, 1.83848739,
                              0.38560664, 0.78341256, 0.94136179],
                          [2.88618565,  2.55316663, 0.98124337, 0.63592154, 1.80029877, 0.29067589]])
instance_bboxes = np.array([[1.48129511, 3.52074146, 1.85652947, 1.74445975, 0.23195696, 0.57235193],
                            [2.90395617, -3.48033905, 1.52682471,
                                0.66077662, 0.17072392, 0.67153597],
                            [1.14655101,  2.1986711,  0.61649567,
                                0.54184127, 2.53463078, 1.21447623],
                            [3.07075286, -2.18329048, 0.43014488,
                                0.4466958, 0.67386925, 0.90444988],
                            [2.72518206, 0.63453132, 0.18862616,
                                0.37375331, 0.36712757, 0.38526765],
                            [2.99394798, 0.60524988, 0.33328897,
                                0.45761609, 0.41525924, 0.64633226],
                            [-1.57269239, -2.9844265,  0.12791303,
                                0.24538207, 0.49648976, 0.27230513],
                            [-0.39350641, 1.60138261, 0.43618175,
                                2.8234272, 2.31575942, 0.91600794],
                            [2.79179931, 1.28866601, 0.91196638,
                                0.93661404, 1.2307744, 1.81276715],
                            [-0.85693234, 2.1348896, 0.22876281,
                                1.11517799, 0.66755795, 0.48457837],
                            [-3.17430639, -2.04097152, 0.25359929,
                                0.49028802, 1.06022966, 0.5020653],
                            [2.9605794, -3.0664463, 0.35337168, 0.70109391, 0.57229543, 0.74308586]])
corners = []
# Calculate and print corners for each bounding box
for bbox in pseudo_bboxes:
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
    actor.GetProperty().SetColor(1, 1, 1)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(2)
    renderer.AddActor(actor)
    corners = []
    # Calculate and print corners for each bounding box
    for bbox in instance_bboxes:
        center = bbox[:3]
        dimensions = bbox[3:6]
        corner = get_box_corners(center, dimensions)
        corners.append(corner)
    bbox = corners

    # dynamic
    pre_bbox_dynamic = a = bbox
# Loop through each matching segment group, create bounding boxes, and add to renderer
for bbox_id in pre_bbox_dynamic:

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
    actor.GetProperty().SetColor(0, 0, 1)  # Set bounding box color to red
    actor.GetProperty().SetLineWidth(2)
    renderer.AddActor(actor)


def get_rectangle_corners(rectangle):
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
    corners = [
        center - half_width * width_direction - half_height * height_direction,
        center + half_width * width_direction - half_height * height_direction,
        center + half_width * width_direction + half_height * height_direction,
        center - half_width * width_direction + half_height * height_direction,
    ]

    return corners


# Your rectangles垂直
rectangles = np.array([
    [-3.42985571e+00,  2.32401423e-02,  1.46670750e+00,  9.99906483e-01,
        1.36757243e-02,  0.00000000e+00,  6.90635033e+00,  2.95090045e+00],
    [-1.47600340e-01,  3.46315723e+00,  1.47564875e+00, -3.88029529e-03, -
        9.99992472e-01, 0.00000000e+00,  6.65903242e+00,  2.95090095e+00],
    [3.22508157e+00,  5.33037050e-02,  1.47204425e+00, -9.99919212e-01, -
        1.27110078e-02, 0.00000000e+00,  6.79440917e+00,  2.95090095e+00],
    [9.87556673e-01, -2.71021251e+00,  1.46369050e+00, -9.99626692e-01,
        2.73217319e-02, 0.00000000e+00,  1.32675228e+00,  2.95090045e+00],
    [-5.71738060e-02, -3.38661338e+00,  1.46310300e+00, -1.29302977e-02,
        9.99916400e-01, 0.00000000e+00,  6.65145244e+00,  2.95090045e+00]
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
    actor.GetProperty().SetLineWidth(2)
    renderer.AddActor(actor)


horizontal_quads = [[[-3.38491600e+00, -3.43502269e+00, 2.93587300e+00],
                    [-3.47938580e+00, 3.47067027e+00, 2.94841900e+00],
                     [3.17959491e+00, 3.44481196e+00, 2.95375600e+00],
                     [3.26597802e+00, -3.34903678e+00, 2.94121000e+00]],

                    [[3.18418449e+00, 3.45564388e+00, 2.87800000e-03],
                     [-3.47479496e+00, 3.48150282e+00, -2.45800000e-03],
                     [-3.38032610e+00, -3.42418982e+00, -1.50040000e-02],
                     [3.27056886e+00, -3.33820424e+00, -9.66700000e-03]]]
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
    actor.GetProperty().SetLineWidth(2)
    renderer.AddActor(actor)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create a VTK render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Render the scene
render_window.Render()

# Start interaction
render_window_interactor.Start()
