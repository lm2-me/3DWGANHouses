"""
functions related to loading and saving data

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Methods:
    >>> clean_point_clouds: preprocess point cloud files
	>>> list_of_files: gets list of all files in a directory
    >>> load_point_cloud_file: load a point cloud file into an Open3D point cloud object
    >>> save_list_to_json: save a list as a .json file
    >>> ptcld_label_by_class: add label to array based on the class in a point cloud
    >>> ptcld_color_by_class: color each point in a point cloud based on the class
    >>> get_color_norm: get a color to use based on a list
    >>> remove_pts_by_label: remove points based on a list of labels
    >>> remove_exterior_points: remove points that represent exterior geometry
    >>> change_labels: change labels from one value to another
    >>> ptcld_scale_fortest: scale the point cloud, scale is set in function
    >>> bbox_height: get the height of a bounding box
    >>> clone_pc: duplicate a point cloud into a new, independent point cloud Open3D object


"""

import open3d as o3d
import numpy as np
import json
import os

def clean_point_clouds(path, save_location):

    # 3:"vehicle", 5:"plant_tree", 8:"furniture", 9:"ground_grass", 17:"fence", 18:"pond_pool", 19:"corridor_path", 20:"balcony/patio", 23:"road", 24:"entrance_gate"
    labels_to_remove = [3, 5, 8, 9, 17, 18, 19, 20, 23, 24, 31]
    
    #! Update which labels are kept in this list
    # 01 'wall', 02 'window', 04 'roof', 06 'door'
    # wall is excluded from list because all other points will be 'walls'
    keep_labels = [2, 4, 6]
    load_count = 0

    all_files = list_of_files(path)

    for file in all_files:
        load_count += 1
        if file.endswith('.ply') and 'voxel' not in file:
            name = file.replace('.ply','')

            name, file_name, pcd, label_array, color_array = load_point_cloud_file(path, file)

            #REMOVE EXTERIOR POINTS
            updated_pc, bbox, updated_labels, updated_colors = remove_exterior_points(pcd, label_array, color_array)
            updated_pc_1, updated_labels_1, updated_colors_1 = remove_pts_by_label(labels_to_remove, updated_pc, updated_labels, updated_colors)
            #vis.view_pc_and_bbox(pcd, bbox, file, save=save, folder=save_location+'/images/withbbox/')

            #REMOVE UNNECESSARY LABELS
            updated_pc_2, updated_labels_2, updated_colors_2 = change_labels(updated_pc_1, updated_labels_1, updated_colors_1, keep_labels)
            #vis.view_pc_from_ptcld(updated_pc_2, file, save=save, folder=save_location+'/images/editedlabels/')

            #SCALE POINT CLOUD
            updated_pc_3 = ptcld_scale_fortest(updated_pc_2, updated_colors_2)

            save_list_to_json(save_location, name,  updated_labels_2)
            o3d.io.write_point_cloud(save_location + '/' + file, updated_pc_3)

    print('Cleaned {} files.'.format(load_count))

def list_of_files(path_loc):
    all_files = [f for f in os.listdir(path_loc) if os.path.isfile(os.path.join(path_loc, f))]
    return all_files

def load_point_cloud_file(file_path, file):
    label_file = file.replace('.ply', '.json')
    # Load the point cloud file
    label_array = ptcld_label_by_class(file_path + '/' + label_file)
    color_array = ptcld_color_by_class(file_path + '/' + label_file, True)
    pcd = o3d.io.read_point_cloud(file_path+'/'+file)
    pcd.colors = o3d.utility.Vector3dVector(color_array)

    name = file.replace('.ply','')

    return name, file, pcd, label_array, color_array

def save_list_to_json(filepath, filename, list_to_save):
    if '.' in filename: name = filename.replace('.ply','.json')
    else: name = filename + '.json'
    location = filepath + '/' + name

    json_list = {}
    for (i, v) in enumerate(list_to_save):
        json_list[i] = int(v)

    with open(location, 'w') as f:
        json.dump(json_list, f)

def ptcld_label_by_class(path):

    with open(path) as json_file:
        data = json.load(json_file)
    
    label_array = []

    for key in data:
        label_array.append(data[key])

    label_array_np = np.array(label_array)
    return label_array_np

def ptcld_color_by_class(path, norm=True):
    with open(path) as json_file:
        data = json.load(json_file)
    
    color_array = []

    if norm:
        for key in data:
            color = get_color_norm(data[key])
            color_array.append(color)

    color_array_np = np.array(color_array)
    return color_array_np

def get_color_norm(index):
    colors_1 = [(0.9019607843137255, 0.09803921568627451, 0.29411764705882354), (0.23529411764705882, 0.7058823529411765, 0.29411764705882354), (1.0, 0.8823529411764706, 0.09803921568627451), (0.2627450980392157, 0.38823529411764707, 0.8470588235294118), (0.9607843137254902, 0.5098039215686274, 0.19215686274509805), (0.5686274509803921, 0.11764705882352941, 0.7058823529411765), (0.25882352941176473, 0.8313725490196079, 0.9568627450980393), (0.9411764705882353, 0.19607843137254902, 0.9019607843137255), (0.7490196078431373, 0.9372549019607843, 0.27058823529411763), (0.9803921568627451, 0.7450980392156863, 0.8313725490196079), (0.27450980392156865, 0.6, 0.5647058823529412), (0.8627450980392157, 0.7450980392156863, 1.0), (0.6039215686274509, 0.38823529411764707, 0.1411764705882353), (1.0, 0.9803921568627451, 0.7843137254901961), (0.5019607843137255, 0.0, 0.0), (0.6666666666666666, 1.0, 0.7647058823529411), (0.5019607843137255, 0.5019607843137255, 0.0), (1.0, 0.8470588235294118, 0.6941176470588235), (0.0, 0.0, 0.4588235294117647), (0.6627450980392157, 0.6627450980392157, 0.6627450980392157), (0.6, 0.6, 0.6), (0.0, 0.0, 0.0)]
    colors_2 = [(0.6980392156862745, 0.0, 0.0), (0.6666666666666666, 1.0, 0.0), (0.0, 0.0, 1.0), (0.4, 0.0, 0.26666666666666666), (0.21176470588235294, 0.7019607843137254, 0.5372549019607843), (0.12156862745098039, 0.12156862745098039, 0.4), (0.0, 0.30196078431372547, 0.2), (0.8, 0.23921568627450981, 0.611764705882353), (0.0, 0.0, 0.2), (0.23921568627450981, 0.23921568627450981, 0.8)]

    colors = colors_1 + colors_2
    return colors[index]

def remove_pts_by_label(labels_to_remove, pcd, labels, color_array):

    new_points = []
    new_labels = []
    new_colors = []

    points = np.asarray(pcd.points)

    cleaned_pc = o3d.geometry.PointCloud()

    for i, label in enumerate(labels):
        
        if not labels_to_remove.count(label) > 0:
            new_points.append(points[i])
            new_labels.append(label)
            new_colors.append(color_array[i])
            modified_mesh = True
    
    cleaned_pc.points.extend(np.array(new_points))
    cleaned_pc.colors = o3d.utility.Vector3dVector(np.array(new_colors))

    return cleaned_pc, new_labels, new_colors

def remove_exterior_points(pcd, labels, color_array):
    new_points = []
    new_labels = []
    new_colors = []

    low_point = []

    wall_points = []

    points = np.asarray(pcd.points)

    cleaned_pc = o3d.geometry.PointCloud()
    wall_pc = o3d.geometry.PointCloud()

    for i, label in enumerate(labels):
        # 1:"wall" 2:"window" 4:"roof" 6:"door" 7:"tower_steeple" 15:"chimney" 16:"ceiling" 22:"dome"
        desired_labels = [1, 2, 4, 6, 7, 15, 16, 22]
        if desired_labels.count(label) > 0:
            wall_points.append(points[i])
    
    wall_pc.points.extend(np.array(wall_points))
    bbox = wall_pc.get_axis_aligned_bounding_box()
    bbox.color = (0, 0, 1)

    bbox_max = bbox.get_max_bound()
    bbox_min = bbox.get_min_bound()
    #print('min {} and max {}'.format(bbox_min, bbox_max))

    for i, point in enumerate(points):
        if (point[0] >= bbox_min[0] and 
            point[0] <= bbox_max[0] and 
            point[2] >= bbox_min[2] and
            point[2] <= bbox_max[2] and 
            point[1] >= bbox_min[1] and
            point[1] <= bbox_max[1]):

            new_points.append(point)
            new_labels.append(labels[i])
            new_colors.append(color_array[i])
        
            if (point[2] <= 0 and 
                point[2] >= bbox_min[2]):
                low_point.append(point)
    
    cleaned_pc.points.extend(np.array(new_points))
    cleaned_pc.colors = o3d.utility.Vector3dVector(np.array(new_colors))

    return cleaned_pc, bbox, new_labels, new_colors

def change_labels(pcd, labels, color_array, keep_labels):
    new_labels = []
    new_colors = []

    for i, label in enumerate(labels):
        if label in keep_labels:
            new_labels.append(label)
            new_colors.append(color_array[i])
        else:
            new_labels.append(1)
            new_colors.append(get_color_norm(1))

    pcd.colors = o3d.utility.Vector3dVector(np.array(new_colors))

    return pcd, new_labels, new_colors
    
def ptcld_scale_fortest(pcd, color_array):

    scaled_pcd = clone_pc(pcd, color_array)

    height = bbox_height(pcd)
    scale = 1.02 / height
    scaled_pcd.scale(scale, (0,0,0))

    return scaled_pcd

def bbox_height(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_max = bbox.get_max_bound()
    bbox_min = bbox.get_min_bound()
    height = float(bbox_max[1] - bbox_min[1])

    return height

def clone_pc(pcd, color_array = None):
    points = np.asarray(pcd.points)
    new_pc = o3d.geometry.PointCloud()

    new_pc.points.extend(points)
    
    if color_array is not None:        
        new_pc.colors = o3d.utility.Vector3dVector(color_array)

    return new_pc