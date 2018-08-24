from skimage.io import imread
from skimage.transform import resize

import os
import random
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt



def imgs_to_array(img_dir, img_size, resize_method = 'idrid'):
    """
    img_dir: the directory path to the images
    img_size: the target image output size
    resize_method: different resize method designed specifically for different dataset
        - idrid: original data resolution:2848*4288, first trim it to 2848*2848, then resize
    """
    
    img_list = sorted(os.listdir(img_dir))    
    img_list = [f for f in img_list if not f.startswith('.')]

    n_imgs = len(img_list)
    imgs = np.zeros((n_imgs, img_size[0], img_size[1], 3))

    for i, img_path in enumerate(img_list):
        if i % 100 == 0:
            print("images loaded:" , str(i))
        path = img_dir + img_path
        img = imread(path)
        if resize_method == 'idrid':
            img = img[:,540:3388,:]
            img = resize(img, img_size)
        elif resize_method == "REFUGEE":
            img = resize(img, img_size)
        imgs[i] = img
        
    return imgs
        
def center_to_array(center_dir, img_size, resize_method = 'idrid'):
    """
    center_dir: .csv file, representing the center of the images
    img_size: the target image output by the original images
    resize_method: different resize method designed specifically for different dataset, 
                   should be the same as "imgs_to_array" function 
    """
    
    center = pd.read_csv(center_dir)
    center = center.dropna(how='all')
    if resize_method == 'idrid':
        center.iloc[:,1] = center.iloc[:,1] - 540
        center.iloc[:,1] = center.iloc[:,1]*(int(img_size[0])/2848)
        center.iloc[:,2] = center.iloc[:,2]*(int(img_size[1])/2848)
    
    return center.iloc[:,1:3].values
    

def show_img(img, center = None):
    """
    show the images, annotating the center if given
    
    Inputs:
        - imgs: the imgs array, 
        - center: tuple, representing the position of the optic disc center
    """
    plt.figure()
    plt.imshow(img)
    if center:
        plt.scatter(center[0], center[1], s=50, c='red', marker='o')
    plt.show()
    
    
def load_hdf5(infile):
    """ Load the hdf5 file """
    with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
        return f["image"][()]

def write_hdf5(arr,outfile):
    """ 
    Write the hdf5 file
    
    Input:
        - arr: the data going to save as the hdf5 file
        - outfile: the file name
    """
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)