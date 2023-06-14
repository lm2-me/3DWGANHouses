"""
functions related to loading and saving data

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Methods:
	>>> list_of_files: gets list of all files in a directory
    >>> list_of_folders: gets list of all folders in a directory
	>>> point_cloud_to_pickle: generates point cloud information in a dictionary and saves it as a pickle file
	>>> load_model_data: loads and returns the dictionary, numpy array, and point cloud file for each building model
	>>> numpy_load_encoded_array: loads the numpy array for a building model
	>>> load_point_cloud_file: loads the colored point cloud array and label array for a building model and returns point cloud name
	>>> load_model_list: loads the contents of a csv file as a list
	>>> load_ply_file: loads the colored point cloud array and label array for a building model
	>>> _load_dictionary_from_pickle: loads a dictionary from a pickle file
	>>> _load_encoded_array: loads the encoded array as a numpy array
	>>> _load_voxel_grid: loads the open3D voxel grid
	>>> save_dictionary_to_pickle: saves a dictionary to a pickle file
	>>> save_list_of_models: saves list of files in a folder to a csv file
	>>> save_list_to_json: saves a .ply file as a .json file
	>>> save_voxel_grid: loads an open3D voxel grid
	>>> save_encode_array: saves an encoded array as a pickle file
	>>> ptcld_save_to_ply: saves a point cloud as a .ply file
	>>> _save_loss_to_csv: saves loss for that epoch to a csv file (appends existing file if existing file is available)
	>>> remove_files_from_dataset: moves files from one directory to another
"""

from collections import UserDict
from csv import writer
from pathlib import Path
import open3d as o3d
import os
import pickle
import numpy as np
import json
import shutil

import utilities.voxelize as vox


def list_of_files(path_loc):
    all_files = [f for f in os.listdir(path_loc) if os.path.isfile(os.path.join(path_loc, f))]
    return all_files

def list_of_folders(path_loc):
    all_folders = [f.name for f in os.scandir(path_loc) if f.is_dir()]

    return all_folders

def point_cloud_to_pickle(path, range = None, load_list = None):
    pc_dictionary = UserDict()
    all_files = list_of_files(path)
    print('Found {} files in {}'.format(len(all_files), path))

    if range == None: range = len(all_files)
    if load_list == None: count = range
    else: count = len(load_list)

    load_count = 0

    for file in all_files:
        if file.endswith('.ply') and 'voxel' not in file:
            name = file.replace('.ply','')

            if load_count < range and (load_list == None or name in load_list):
                load_count += 1
                # Load the point cloud file
                pcd, label_array, color_array = load_ply_file(path, file)

                typology = ''.join([c for c in name if c.isupper()])
                split = name.split('_')

                if len(split) > 2:
                    type = split[0] + '_' + split[len(split)-2]
                else:
                    type = split[0]
                
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox_max = bbox.get_max_bound()
                bbox_min = bbox.get_min_bound()

                length = float(bbox_max[0] - bbox_min[0])
                width = float(bbox_max[2] - bbox_min[2])
                height = float(bbox_max[1] - bbox_min[1])

                hull, _ = pcd.compute_convex_hull()
                hull_volume = hull.get_oriented_bounding_box().volume()

                pc_dictionary = {
                    "name": name,
                    "orig file name": file,
                    "scale factor": None,
                    "labels": label_array,
                    "colors": color_array,
                    "typology": typology,
                    "building type": type, 
                    "conv hull volume": hull_volume,
                    "bbox volume": length*width*height,
                    "bbox length": length,
                    "bbox width": width,
                    "bbox height": height,
                    "voxel size": None,
                    "num pcd voxels": None,
                    "pcd memory": None
                }

                pickle_save_path = path + '/' + name + '.pkl' 

                save_dictionary_to_pickle(pickle_save_path, pc_dictionary)

    print('Saved dictionary of {} point clouds to pickle file'.format(load_count))

# load functions
def load_model_data(path, file):
    name = file.replace('.ply','')
    dictionary = None
    voxel_grid = None
    encoded_array = None

    if os.path.isfile(path + '/' + name + '.pkl'): dictionary = _load_dictionary_from_pickle(path + '/' + name + '.pkl')
    if os.path.isfile(path + '/' + name + 'voxel.ply'): voxel_grid = _load_voxel_grid(path + '/' + name + 'voxel.ply')
    if os.path.isfile(path + '/' + name + 'encode.npy'): encoded_array = _load_encoded_array(path + '/' + name + 'encode.npy')

    return dictionary, voxel_grid, encoded_array

def numpy_load_encoded_array(path, file):
    path, file = path.decode(), file.decode()

    name = file.replace('.ply','')
    encoded_array = _load_encoded_array(path + '/' + name + 'encode.npy')
    
    return encoded_array.astype(np.float32)

def load_point_cloud_file(file_path, file):
    label_file = file.replace('.ply','_label.json')
    label_path = 'dataset/test_set_onestory_200/'
    # Load the point cloud file
    label_array = vox.ptcld_label_by_class(label_path+label_file)
    color_array = vox.ptcld_color_by_class(label_path+label_file, True)
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.colors = o3d.utility.Vector3dVector(color_array)

    name = file.replace('.ply','')

    return name, file, pcd, label_array, color_array

def load_model_list(filepath, csv_name):
    location = filepath + '/' + csv_name

    with open(location, newline='') as f:
        data = [l.strip() for l in f.readlines()]

    return data

def load_ply_file(path, file):
    # Load the point cloud file
    label_path = path + "/" + file.replace('.ply','.json')
    if Path(label_path).is_file(): 
        label_path = label_path
    else:
        label_path = label_path.replace('.json','_label.json')

    label_array = vox.ptcld_label_by_class(label_path)
    color_array = vox.ptcld_color_by_class(label_path, True)

    
    file_path = path + "/" + file
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.colors = o3d.utility.Vector3dVector(color_array)

    return pcd, label_array, color_array

# internal: load model data
def _load_dictionary_from_pickle(pickle_save_path):
    read = open(pickle_save_path, "rb")
    loaded_dict = pickle.load(read)

    return loaded_dict

def _load_encoded_array(pathname):
    with open(pathname, 'rb') as f:
        data = np.load(f)
    if len(data.shape) == 5:
        return data[0]
    else:
        return data

def _load_voxel_grid(filename):
   voxel_grid = o3d.io.read_voxel_grid(filename, format='auto', print_progress=False)
   return voxel_grid


# save functions
def save_dictionary_to_pickle(pickle_save_path, dictionary): 
    with open(pickle_save_path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_list_of_models(filepath, csv_name):
    all_files = list_of_files(filepath)
    namelist = []

    for file in all_files:
        name = file.replace('.png','')
        namelist.append(str(name))

    csv_folder = 'C:\\TUDelft\\MScThesis\\preprocessing\\csv'
    location = csv_folder + '\\' + csv_name + '.csv'

    with open(location, 'w') as f:
        for name in namelist:
            f.write(name)
            f.write('\n')

def save_list_to_json(filepath, filename, list_to_save):
    if '.' in filename: name = filename.replace('.ply','.json')
    else: name = filename + '.json'
    location = filepath + '/' + name

    json_list = {}
    for (i, v) in enumerate(list_to_save):
        json_list[i] = int(v)

    with open(location, 'w') as f:
        json.dump(json_list, f)

def save_voxel_grid(filename, voxel_grid):
    o3d.io.write_voxel_grid(filename, voxel_grid, write_ascii=False, compressed=False, print_progress=False)

def save_encode_array(pathname, array):
    np.save(pathname, array, allow_pickle=True, fix_imports=True)

def ptcld_save_to_ply(folder, file_name, pcd):

    o3d.io.write_point_cloud(folder + '/' + file_name, pcd)

def _save_loss_to_csv(path, new_row):
    if not Path(path).is_file():
        with open(path, 'w') as file:
            row = ['epoch', 'G loss', 'D real loss', 'D fake loss', 'D total loss', 'D real acc', 'D fake acc', 'update']
            writer_object = writer(file)
            writer_object.writerow(row)
            file.close()
    
    with open(path, 'a') as open_file:
        writer_object = writer(open_file)
        writer_object.writerow(new_row)
        open_file.close()

def remove_files_from_dataset(remove_list, origin_path, remove_path):
    
    for file in os.listdir(origin_path):
        for remove in remove_list:
            if remove in str(file):
                shutil.move(origin_path + '/' + file, remove_path + '/' + file)
