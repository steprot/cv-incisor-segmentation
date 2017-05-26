# -*- coding: utf-8 -*-
""" Estimate model on radiograph
    - automatic initialization of the location of an incisor, before the
    iterative fitting procedure is started. """

import math
import sys
import cv2
import numpy as np
from sklearn.decomposition import PCA
from init import print_boxes_on_teeth

from preprocessing import load_radiographs, preprocess_radiograph

def normalize_image(img):
    """Normalize an image such that it min=0 , max=255 and type is np.uint8
    """
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)
 
 
def project(W, X, mu):
    """Project X on the space spanned by the vectors in W.
    mu is the average image.
    """
    return np.dot(X - mu.T, W)


def reconstruct(W, Y, mu):
    """Reconstruct an image based on its PCA-coefficients Y, the evecs W
    and the average mu.
    """
    return np.dot(Y, W.T) + mu.T

def slide(image, seg, step, window):
    # Maybe change it a bit later if we have time so it would return one element and call it n times
    for y in range(seg[0][1], seg[1][1] - window[1], step) + [seg[1][1] - window[1]]:
        for x in range(seg[0][0], seg[1][0] - window[0], step) + [seg[1][0] - window[0]]:
            yield (x, y, image[y:y + window[1], x:x + window[0]])

def getcut(img,a1,b1,a2,b2):
    
    h, w = img.shape
    #print(img.shape)   
    crp = img[a1 :a2, b1:b2]
    #crp = img[b1:b2, a1 :a2]
    cv2.imshow("cut it damn",crp)
    return crp
    
def best_seg(mean, evecs, image, is_upper, largest_boxes, width, height, show=False):
    
    # ------------------------------------------------
    # THIS REALLY SUCKS AND ALSO DOESNT FREAKING WORK
    # ------------------------------------------------
    """Finds a bounding box around the four upper or lower incisors.
    A sliding window is moved over the given image. The window which matches best
    with the given appearance model is returned.

    Args:
        mean: PCA mean.
        evecs: PCA eigen vectors.
        image: The dental radiograph on which the incisors should be located.
        width (int): The default width of the search window.
        height (int): The default height of the search window.
        is_upper (bool): Wheter to look for the upper (True) or lower (False) incisors.
        jaw_split (Path): The jaw split.

    Returns:
        A bounding box around what looks like four incisors.
        The region of the image selected by the bounding box.

    """
    h, w = image.shape

    # [b1, a1]---------------
    # -----------------------
    # -----------------------
    # -----------------------
    # ----------------[b2 a2]

    if is_upper:
        #b1 = int(w/2 - w/10)
        #b2 = int(w/2 + w/10)
        b1 = int(largest_boxes[0])
        b2 = int(largest_boxes[2])
        a1 = int(largest_boxes[1]) 
        a2 = int(largest_boxes[3])
    else:
        #b1 = int(w/2 - w/12)
        #b2 = int(w/2 + w/12)
        b1 = int(largest_boxes[4])
        b2 = int(largest_boxes[6])
        a1 = int(largest_boxes[5])
        a2 = int(largest_boxes[7])
    search_region = [(b1, a1), (b2, a2)]

    best_score = 100000
    best_score_bbox = [(-1, -1), (-1, -1)]
    best_score_img = np.zeros((width, height))
    for wscale in np.arange(0.8, 1.3, 0.1): # start 0.8 stop 1.3 step 0.1 -- Try different scales for width
        for hscale in np.arange(0.7, 1.3, 0.1): # start 0.7 stop 1.3 step 0.1 -- Try different scales for hight
            winW = int(width * wscale)
            winH = int(height * hscale)
            for (x, y, window) in slide(image, search_region, 36, (winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                reCut = cv2.resize(window, (width, height))
                cv2.imshow('recut', reCut)

                X = reCut.flatten()
                Y = project(evecs, X, mean)
                Xacc = reconstruct(evecs, Y, mean)

                score = np.linalg.norm(Xacc - X)
                print(score)
                if score < best_score:
                    best_score = score
                    best_score_bbox = [(x, y), (x + winW, y + winH)]
                    best_score_img = reCut
                    cv2.imshow('IF recut', reCut)

                #if show:
                #    window = [(x, y), (x + winW, y + winH)]
                #    Plotter.plot_autoinit(image, window, score, jaw_split, search_region, best_score_bbox,
                #                          title="wscale="+str(wscale)+" hscale="+str(hscale))

    return (best_score_bbox)


def estimate(model,toothnr,preprocessed_r,coord):
    """ 
    model - The shape we want to fit.
    toothnr - which incisor are we looking for?
    
    """
    display = False
    
    #if toothnr < 5:
    #    # We calculated the average hight of upper teeth and it was around 315,
    #    # the average width around: 380
    #    width = 380
    #    height = 315 
    #    isupper = True   
    #else:
    #    # We calculated the average hight of upper teeth and it was around 260,
    #    # the average width around: 280
    #    width = 280
    #    height = 260
    #    isupper = False
        
    if toothnr < 5:
        # We calculated the average hight of upper teeth and it was around 315,
        # the average width around: 380
        width = coord[2]-coord[0]
        height = coord[3]-coord[1] 
        isupper = True   
    else:
        # We calculated the average hight of upper teeth and it was around 260,
        # the average width around: 280
        width = coord[6]-coord[4]
        height = coord[7]-coord[5]
        isupper = False
        
    # Create the appearance model for the four upper/lower teeth
    #radiographs = load_radiographs(11,False) 
    data = getcut(preprocessed_r, coord[1], coord[0], coord[3], coord[2])
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
    pca_res.fit(np.asarray(data))
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
    [(a, b), (c, d)] = best_seg(mean, eigen_vec, data, isupper, coord, width, height, False)
    
    print([a, b, c, d])
    print_boxes_on_teeth([a, b, c, d], preprocessed_r)
    # TO BE CONTINUED!!!!!

    