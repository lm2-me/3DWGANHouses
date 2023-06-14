"""
main function to train the network

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Methods:
    >>> main: runs wgan17R file
"""

print(__file__)

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import wganv17R as gan

def main():
    #! update path if you are using a different dataset
    ptcld_path_preprocessed = 'sample_files/dataset'

    gan.train_gan(ptcld_path_preprocessed, 'wgan17R_test', 160*160*80)
        

if __name__ == '__main__':
    main()