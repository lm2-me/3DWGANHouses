"""
functions related to loading and saving data of GANs

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Methods:
	>>> list_of_files: returns list of files in directory
    >>> _save_loss_to_csv: saves the loss data to a csv file
    >>> save_encode_array: saves the encoded matrix to a numpy file
    >>> load_data_for_gan_individually: loads training data based on the buffer size
    >>> numpy_load_encoded_array: decodes and loads numpy array with _load_encoded_array
    >>> _load_encoded_array: loads the encoded matrix from a numpy file 
"""


from csv import writer
from pathlib import Path
import os
import numpy as np
import tensorflow as tf

def list_of_files(path_loc):
    all_files = [f for f in os.listdir(path_loc) if os.path.isfile(os.path.join(path_loc, f))]
    return all_files

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

def save_encode_array(pathname, array):
    np.save(pathname, array, allow_pickle=True, fix_imports=True)

def load_data_for_gan_individually(path):
    
    all_files = list_of_files(path)
    print('Found {} files in {}'.format(len(all_files), path))

    load_count = 0
    filtered_filenames = []

    for file in all_files:
        if file.endswith('.ply') and 'voxel' not in file:
            name = file.replace('.ply','')

            load_count += 1
            filtered_filenames.append(file)

    filename_dataset = tf.data.Dataset.from_tensor_slices(filtered_filenames) \
        .shuffle(len(filtered_filenames), reshuffle_each_iteration=True)
    
    geometry_dataset = filename_dataset.map(lambda file:
        tf.numpy_function(numpy_load_encoded_array, [path, file], tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE)

    print('Loaded {} file(s)'.format(load_count))

    return geometry_dataset

def numpy_load_encoded_array(path, file):
    if hasattr(path, 'decode'):
        path = path.decode()
    if hasattr(file, 'decode'):
        file = file.decode()

    name = file.replace('.ply','')

    encoded_array = _load_encoded_array(path + '/' + name + 'encode.npy')
    
    return encoded_array.astype(np.float32)

def _load_encoded_array(pathname_array):
    with open(pathname_array, 'rb') as f:
        data = np.load(f)
    if len(data.shape) == 5:
        return data[0]
    else:
        return data