# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
from scipy.ndimage import morphology

def load_radiographs(number_samples, testing):
    '''
    Load the radiograph images and returns an array with the requested radiographs as 3-channel color images,
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
    Enhances a dental x-ray image by:
        making it in greyscale. 
        applying the median filter, applying a bilateral filter,
        combining the top- and bottom-hat transformations, applying CLAHE
    Parameters:
        img; the radiograph 
        The enhanced radiograph as a grayscale image.
    '''
    #cv2.imshow('Initial Radiograph', img)
    #cv2.waitKey(0)
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #img = adaptive_median(img, 3, 5)
    
    # using only the median filter instead of the adaptive median filter. 
    img = cv2.medianBlur(img, 3)

    img = bilateral_filter(img)
    
    img_top = top_hat_transform(img)
    img_bottom = bottom_hat_transform(img)
    img = cv2.add(img, img_top)
    img = cv2.subtract(img, img_bottom)

    img = clahe(img)
    #cv2.imshow('Edges', img)
    #cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return img
    
def bilateral_filter(img):
    '''
    Applies a bilateral filter to the given image.
    Parameters:
        img; a grayscale image of the radiograph after median filter
    '''
    return cv2.bilateralFilter(img, 9, 175, 175)
    
def top_hat_transform(img):
    '''
    Calculates the top-hat transformation of a given image.
    Parameters:
        img; a grayscale image of the radiograph after median filter and bilateral filter
    '''
    return morphology.white_tophat(img, size=400)


def bottom_hat_transform(img):
    '''
    Calculates the bottom-hat transformation of a given image.
    Parameters:
        img; a grayscale image of the radiograph after median filter and bilateral filter
    '''
    return morphology.black_tophat(img, size=80)
    
def clahe(img): # Contrast Limited Adaptive Histogram Equalization
    '''
    Creates a CLAHE object and applies it to the given image.
    Parameters:
        img; a grayscale image of the radiograph already processed
    '''
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe_obj.apply(img)


def togradient_sobel(img):
    '''
    Applies the Sobel Operator and returns the image with the edges. 
    Parameters:
        img; the preprocessed greyscale image.
    '''
    img = cv2.GaussianBlur(img,(3,3),0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    
    img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return img
    #return cv2.addWeighted(sobelx, 0.1, sobely, 0.1, 0)
    
def adaptive_median(img, window, threshold):
    '''
    Applies an adaptive median filter to the image. This is essentially a
    despeckling filter for grayscale images.
    Args:
        image: The source image, as a numpy array
        window: Sets the filter window size (must be a scalar between 1 and 5).
                Window size (ws) is defined as W = 2*ws + 1 so that W = 3 is a
                3x3 filter window.
        threshold: Sets the adaptive threshold (0=normal median behavior).
                    Higher values reduce the "aggresiveness" of the filter.
    '''
    #img = image.copy()Top
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # set filter window and image dimensions
    W = 2*window + 1
    ylength, xlength = img.shape
    vlength = W*W

    # create 2-D image array and initialize window
    filter_window = np.array(np.zeros((W, W)))
    target_vector = np.array(np.zeros(vlength))
    pixel_count = 0

    print(img.shape)
    # loop over image with specified window W
    for y in range(window, ylength - (window + 1)):
        for x in range(window, xlength - (window + 1)):
            #if x%500 == 0 and y%500 == 0:
            #    print(x,y)
            # populate window, sort, find median
            filter_window = img[y - window:y + window + 1, x - window:x + window + 1]
            target_vector = np.reshape(filter_window, ((vlength),))
            # internal sort
            median = med(target_vector, vlength)
            # check for threshold
            if not threshold > 0:
                img[y, x] = median
                pixel_count += 1
            else:
                scale = np.zeros(vlength)
                for n in range(vlength):
                    scale[n] = abs(target_vector[n] - median)
                    #if n%20 == 0:
                    #    print(x, scale[n], 'target_vector[n]',  target_vector[n], 'median',  median)
                scale = np.sort(scale)
                Sk = 1.4826 * (scale[vlength/2])
                if abs(img[y, x] - median) > (threshold * Sk):
                    img[y, x] = median
                    pixel_count += 1
    
    print(img.shape)
    print(pixel_count, "pixel(s) filtered out of", xlength*ylength)
    return img

def med(target_array, array_length):
    '''
    (for the adaptive median)
    Computes the median of a sublist
    Parameters: 
        target_array; the list from which return the median
        array_length; length of the list.
    '''
    sorted_array = np.sort(target_array)
    median = sorted_array[array_length/2]
    return median
    
def resize(image, width, height):
    '''
    (for the adaptive median)
    Resizes the given image to the given width and height. The scaling factor is returned. 
    Parameters:
        image: The radiograph to resize.
        width (int): The new width for the image.
        height (int): The new height for the image.
    '''
    #find minimum scale to fit image on screen
    scale = min(float(width) / image.shape[1], float(height) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale))), scale
    