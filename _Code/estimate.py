""" Estimate model on radiograph
    - automatic initialization of the location of an incisor, before the
    iterative fitting procedure is started. """

import math
import sys
import cv2
import numpy as np
from sklearn.decomposition import PCA

from preprocessing import load_radiographs, preprocess_radiograph

def build_segment():
    
    # todooo
    
    return 0

def load_database(radiographs, is_upper, newwidth, newheight):
    """Extracts the ROI's, selected in ``create_database``, from the radiographs
    and scales the image regions to a uniform size.

    Args:
        radiographs: A list with dental radiographs for which a ROI was selected
            in ``create_database``.
        is_upper: whether to load the database for the upper/lower teeth.

    Returns:
        Every image is cropped to the ROI and resized to rewidth*reheight.

    """
    newwidth=500
    newheight=500
    smallImages = np.zeros((14, newwidth * newheight))
    try:
        if is_upper:
            four_incisor_bbox = np.load('Models/upper_incisor_model.npy')
        else:
            four_incisor_bbox = np.load('Models/lower_incisor_model.npy')
    except IOError:
        sys.exit("Create a database first!")

    for ind, radiograph in enumerate(radiographs):
        [(x1, y1), (x2, y2)] = four_incisor_bbox[ind-1]
        segmented = radiograph[y1:y2, x1:x2]
        result = cv2.resize(segmented, (newwidth, newheight), interpolation=cv2.INTER_NEAREST)
        segmented[ind-1] = result.flatten()

    return segmented

def slide(image, seg, step, window):
    # Maybe change it a bit later if we have time so it would return one element and call it n times
    for y in range(seg[0][1], seg[1][1] - window[1], step) + [seg[1][1] - window[1]]:
        for x in range(seg[0][0], seg[1][0] - window[0], step) + [seg[1][0] - window[0]]:
            yield (x, y, image[y:y + window[1], x:x + window[0]])
#
#def best_seg(mean, evecs, image, width, height, is_upper, show):
#    
#    # ------------------------------------------------
#    # THIS REALLY SUCKS AND ALSO DOESNT FREAKING WORK
#    # ------------------------------------------------
#    """Finds a bounding box around the four upper or lower incisors.
#    A sliding window is moved over the given image. The window which matches best
#    with the given appearance model is returned.
#
#    Args:
#        mean: PCA mean.
#        evecs: PCA eigen vectors.
#        image: The dental radiograph on which the incisors should be located.
#        width (int): The default width of the search window.
#        height (int): The default height of the search window.
#        is_upper (bool): Wheter to look for the upper (True) or lower (False) incisors.
#        jaw_split (Path): The jaw split.
#
#    Returns:
#        A bounding box around what looks like four incisors.
#        The region of the image selected by the bounding box.
#
#    """
#    h, w = image.shape
#
#    # [b1, a1]---------------
#    # -----------------------
#    # -----------------------
#    # -----------------------
#    # ---------------[b2, a2]
#
#    if is_upper:
#        b1 = int(w/2 - w/10)
#        b2 = int(w/2 + w/10)
#        a1 = int(np.max(jaw_split.get_part(b1, b2), axis=0)[1]) - 350
#        a2 = int(np.max(jaw_split.get_part(b1, b2), axis=0)[1])
#    else:
#        b1 = int(w/2 - w/12)
#        b2 = int(w/2 + w/12)
#        a1 = int(np.min(jaw_split.get_part(b1, b2), axis=0)[1])
#        a2 = int(np.min(jaw_split.get_part(b1, b2), axis=0)[1]) + 350
#
#    search_region = [(b1, a1), (b2, a2)]
#
#    best_score = 100000
#    best_score_bbox = [(-1, -1), (-1, -1)]
#    best_score_img = np.zeros((500, 400))
#    for wscale in np.arange(0.8, 1.3, 0.1):
#        for hscale in np.arange(0.7, 1.3, 0.1):
#            winW = int(width * wscale)
#            winH = int(height * hscale)
#            for (x, y, window) in slide(image, search_region, step_size=36, window_size=(winW, winH)):
#                # if the window does not meet our desired window size, ignore it
#                if window.shape[0] != winH or window.shape[1] != winW:
#                    continue
#
#                reCut = cv2.resize(window, (width, height))
#
#                X = reCut.flatten()
#                Y = project(evecs, X, mean)
#                Xacc = reconstruct(evecs, Y, mean)
#
#                score = np.linalg.norm(Xacc - X)
#                if score < best_score:
#                    best_score = score
#                    best_score_bbox = [(x, y), (x + winW, y + winH)]
#                    best_score_img = reCut
#
#                if show:
#                    window = [(x, y), (x + winW, y + winH)]
#                    Plotter.plot_autoinit(image, window, score, jaw_split, search_region, best_score_bbox,
#                                          title="wscale="+str(wscale)+" hscale="+str(hscale))
#
    #return (best_score_bbox)


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
    #radiographs = load_radiographs(11,False) 
    data = load_database(preprocessed_r, upper, width, height)
    pca_res = PCA(n_components=8) 
                           # Instead of setting th nr of components, we make sure 
                           # that we capture 99% of the variance 
                           # this gives eigenvectors with different length! for each incisor so 
                           # it was better to set the nr to a fixed value, 8 cover around 99%
        # From pca_res we can obtain :
            # components_
            # explained_variance_
            # explained_variance_ratio_
            # components_
            # mean_
            # n_components_
    pca_res.fit(data)
    eigen_vec= pca_res.components_
    mean  = pca_res.mean_
    # -----
    # Visualize the appearance model
    # cv2.imshow('img',np.hstack( (mean.reshape(height,width),
    #                              normalize(eigen_vec[:,0].eigen_vec(height,width)),
    #                              normalize(eigen_vec[:,1].eigen_vec(height,width)),
    #                              normalize(eigen_vec[:,2].eigen_vec(height,width)))
    #                            ).astype(np.uint8))
    # cv2.waitKey(0)
    
    #------
    # Find the region of the radiograph that matches best with the appearance model
    img = preprocess_radiograph(img) 
    #[(a, b), (c, d)]= best_seg(mean, eigen_vec, img, width, height, upper, False)
    # TO BE CONTINUED!!!!!
    