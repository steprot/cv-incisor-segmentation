# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from scipy.ndimage import morphology


def load_radiographs(number_samples,testing):
    '''
        Load the radiograph images
        Returns:
            An array with the requested radiographs as 3-channel color images,
            ordered the same as the given indices.
    '''
    images = []
    if testing:
        directory = '../_Data/Radiographs/test.tif'
        dir_path = os.path.join(os.getcwd(), directory)
        # Read the image and save it in images
        images = cv2.imread(dir_path)
    else:
        i = 0        
        while i < number_samples:
            directory = '../_Data/Radiographs/' + str("%02d" % (i+1)) + '.tif'
            dir_path = os.path.join(os.getcwd(), directory)
            # Read the image and save it in images
            images.append(cv2.imread(dir_path))
            # Go to the next sample 
            i += 1
        
    return images


def preprocess_radiograph(img):
    '''
        Enhances a dental x-ray image by
            1. applying a bilateral filter,
            2. combining the top- and bottom-hat transformations
            3. applying CLAHE
        Args:
            img: A dental x-ray image.
        Returns:
            The enhanced radiograph as a grayscale image.
    '''
    cv2.imshow('Initial Radiograph', img)
    cv2.waitKey(0)
    
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # bisogna implementare ancora questa funzione. 
    # è però implementata precendemente e le immagini sono poi salvate da qualche parte. 
    # problema nel trovare un'implementazione decente di questa cosa. 
    
    # img = adaptive_median(img, 3, 5) # WHY IS IT COMMENTED?
    
    img = bilateral_filter(img)
    #cv2.imshow('Bilateral Filter Radiograph', img)
    #cv2.waitKey(0)
    
    img_top = top_hat_transform(img)
    #cv2.imshow('Top Hat Filter Radiograph', img)
    #cv2.waitKey(0)
    img_bottom = bottom_hat_transform(img)
    #cv2.imshow('Bottom Hat Filter Radiograph', img)
    #cv2.waitKey(0)
    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)
    #cv2.imshow('Applied Filter Radiograph', img)
    #cv2.waitKey(0)

    img = clahe(img)
    #cv2.imshow('Clahe Filter Radiograph', img)
    #cv2.waitKey(0)
    
    # Finding the edges 
    img = togradient_sobel(img)
    cv2.imshow('Sobel Filter Radiograph', img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return img
    
def bilateral_filter(img):
    '''
        Applies a bilateral filter to the given image.
        This filter is highly effective in noise removal while keeping edges sharp.
        Args: img: A grayscale dental x-ray image.
        Returns: The filtered image.
    '''
    return cv2.bilateralFilter(img, 9, 175, 175)
    
def top_hat_transform(img):
    '''
        Calculates the top-hat transformation of a given image.
        This transformation enhances the brighter structures in the image.
        Args: img: A grayscale dental x-ray image.
        Returns: The top-hat transformation of the input image.
    '''
    return morphology.white_tophat(img, size=400)


def bottom_hat_transform(img):
    '''
        Calculates the bottom-hat transformation of a given image.
        This transformation enhances the darker structures in the image.
        Args: img: A grayscale dental x-ray image.
        Returns: The top-hat transformation of the input image.
    '''
    return morphology.black_tophat(img, size=80)
    
def clahe(img): # Contrast Limited Adaptive Histogram Equalization
    '''
        Creates a CLAHE object and applies it to the given image.
        Args: img: A grayscale dental x-ray image.
        Returns: The result of applying CLAHE to the given image.
    '''
    
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)


def togradient_sobel(img):
    '''
        Applies the Sobel Operator.
        Args: img: A grayscale dental x-ray image.
        Returns: An image with the detected edges bright on a darker background.
    '''
    img = cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    
#def adaptive_median(image_array, window, threshold):
#    """Applies an adaptive median filter to the image. This is essentially a
#    despeckling filter for grayscale images.
#
#    Args:
#        image_array: The source image, as a numpy array
#        window: Sets the filter window size (must be a scalar between 1 and 5).
#                Window size (ws) is defined as W = 2*ws + 1 so that W = 3 is a
#                3x3 filter window.
#        threshold: Sets the adaptive threshold (0=normal median behavior).
#                    Higher values reduce the "aggresiveness" of the filter.
#
#    Returns:
#        The filtered image.
#
#    .. _Based on:
#        https://github.com/sarnold/adaptive-median/blob/master/adaptive_median.py
#
#    """
#    image_array = image_array.copy()
#    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
#
#
#    def med(target_array, array_length):
#        """Computes the median of a sublist.
#        """
#        sorted_array = np.sort(target_array)
#        median = sorted_array[array_length/2]
#        return median
#
#    # set filter window and image dimensions
#    W = 2*window + 1
#    ylength, xlength = image_array.shape
#    vlength = W*W
#
#    # create 2-D image array and initialize window
#    filter_window = np.array(np.zeros((W, W)))
#    target_vector = np.array(np.zeros(vlength))
#    pixel_count = 0
#
#    # loop over image with specified window W
#    for y in range(window, ylength-(window+1)):
#        for x in range(window, xlength-(window+1)):
#            # populate window, sort, find median
#            filter_window = image_array[y-window:y+window+1, x-window:x+window+1]
#            target_vector = np.reshape(filter_window, ((vlength),))
#            # internal sort
#            median = med(target_vector, vlength)
#            # check for threshold
#            if not threshold > 0:
#                image_array[y, x] = median
#                pixel_count += 1
#            else:
#                scale = np.zeros(vlength)
#                for n in range(vlength):
#                    scale[n] = abs(target_vector[n] - median)
#                scale = np.sort(scale)
#                Sk = 1.4826 * (scale[vlength/2])
#                if abs(image_array[y, x] - median) > (threshold * Sk):
#                    image_array[y, x] = median
#                    pixel_count += 1
#
#    print(pixel_count, "pixel(s) filtered out of", xlength*ylength)
#    return image_array
