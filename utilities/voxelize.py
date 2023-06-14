"""
functions related to voxelizing models

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Methods:
    >>> ptcld_color_by_class: assign colors to each point in a point cloud based on the class
    >>> ptcld_label_by_class: assign labels to each point in a point cloud based on the class
"""

import utilities.ganutilities as utils
import numpy as np
import json

def ptcld_color_by_class(path, norm=False):
    with open(path) as json_file:
        data = json.load(json_file)
    
    color_array = []

    if norm:
        for key in data:
            color = utils.get_color_norm(data[key])
            color_array.append(color)

    else:
        for key in data:
            color = utils.get_color(data[key])
            color_array.append(color)

    color_array_np = np.array(color_array)
    return color_array_np

def ptcld_label_by_class(path):

    with open(path) as json_file:
        data = json.load(json_file)
    
    label_array = []

    for key in data:
        label_array.append(data[key])

    label_array_np = np.array(label_array)
    return label_array_np
