""" Estimate model on radiograph
    - automatic initialization of the location of an incisor, before the
    iterative fitting procedure is started. """

import math
import sys
import cv2
import numpy as np

from preprocessing import load_radiographs, preprocess_radiograph

def build_segment():
    
    # todooo
    
    return 0

def load_database(radiographs, is_upper, newwidth=500, newheight=500, radiographs):
    """Extracts the ROI's, selected in ``create_database``, from the radiographs
    and scales the image regions to a uniform size.

    Args:
        radiographs: A list with dental radiographs for which a ROI was selected
            in ``create_database``.
        is_upper: whether to load the database for the upper/lower teeth.

    Returns:
        Every image is cropped to the ROI and resized to rewidth*reheight.

    """
    smallImages = np.zeros((14, rewidth * reheight))
    try:
        if is_upper:
            four_incisor_bbox = np.load('Models/upper_incisor_model.npy')
        else:
            four_incisor_bbox = np.load('Models/lower_incisor_model.npy')
    except IOError:
        sys.exit("Create a database first!")

    for ind, radiograph in enumerate(radiographs):
        [(x1, y1), (x2, y2)] = four_incisor_bbox[ind-1]
        get_segment = radiograph[y1:y2, x1:x2]
        result = cv2.resize(get_segment, (newwidth, newheight), interpolation=cv2.INTER_NEAREST)
        segmented[ind-1] = result.flatten()

    return segmented

def estimate(model,toothnr,preprocessed_r):
    """ 
    model - The shape we want to fit.
    toothnr - which incisor are we looking for?
    
    """
    display = False
    img = load_radiographs(1,True)
    
    if toothnr < 5:
        # We calculated the average hight of upper teeth and it was around 315,
        # the average width around: 380
        width = 380
        height = 315 
        upper = True   
    else:
        # We calculated the average hight of upper teeth and it was around 260,
        # the average width around: 280
        width = 280
        height = 260
        upper = False
        
    # Create the appearance model for the four upper/lower teeth
    radiographs = load_radiographs(11,False) 
    data = load_database(radiographs, upper, width, height,preprocessed_r)
    #[_, evecs, mean] = pca(data, 5)
    