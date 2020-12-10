import time
import os

import cv2
import argparse
import numpy as np
from scipy import signal
from math import ceil, floor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Credit: Project 1 CS 445
def gaussian_kernel(sigma, kernel_half_size):
    '''
    Inputs:
        sigma = standard deviation for the gaussian kernel
        kernel_half_size = recommended to be at least 3*sigma
    
    Output:
        Returns a 2D Gaussian kernel matrix
    '''
    window_size = kernel_half_size*2+1
    gaussian_kernel_1d = signal.gaussian(window_size, std=sigma).reshape(window_size, 1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    gaussian_kernel_2d /= np.sum(gaussian_kernel_2d) 

    return gaussian_kernel_2d

def get_patch_indices(image, i, j, patch_size):
    """
    @brief      Return a patch of size patch_size that is within the bounds of the image
                
    @param      image      image to extract patch
    @param      i          patch height index
    @param      j          patch width index
    @param      patch_size tuple containing desired patch dimensions
    
    @return     tuple of 4 elements: (height start, height end, width start, width end)
    """
    if i >= image.shape[0]-patch_size and j >= image.shape[1]-patch_size:
        indices = (image.shape[0]-patch_size, image.shape[0], image.shape[1]-patch_size, image.shape[1])
    elif i >= image.shape[0]-patch_size:
        indices = (image.shape[0]-patch_size, image.shape[0], j, j+patch_size)
    elif j >= image.shape[1]-patch_size:
        indices = (i, i+patch_size, image.shape[1]-patch_size, image.shape[1])
    else:
        indices = (i, i+patch_size, j, j+patch_size)
    return indices