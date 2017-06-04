# -*- coding: utf-8 -*-
'''
Estimate model on radiograph
- automatic initialization of the location of an incisor, before the
iterative fitting procedure is started.
'''
import cv2
import numpy as np
import pickle
import os
 
def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''   
    Xnew = X - mu.T
    return np.dot(Xnew, W)

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the evecs W
    and the average mu.
    '''
    return np.dot(Y, W.T) + mu.T

def slide(image, seg, step, window):
    '''
    Apply sliding window on the image
    '''
    # Maybe change it a bit later if we have time so it would return one element and call it n times
    for y in range(seg[0][1], seg[1][1] - window[1], step) + [seg[1][1] - window[1]]:
        for x in range(seg[0][0], seg[1][0] - window[0], step) + [seg[1][0] - window[0]]:
            yield (x, y, image[y:y + window[1], x:x + window[0]])

def getcut(img, a1, b1, a2, b2):
    '''
    Cut image segment from a1b1 to a2b2
    '''
    h, w = img.shape
    #print(img.shape)   
    crp = img[a1 :a2, b1:b2]
    #crp = img[b1:b2, a1 :a2]
    cv2.imshow("Cropped image", crp)
    return crp
 
def cut_radiographs(radiographs, is_upper, four_incisor_bbox, rewidth, reheight, number_samples):
    '''
    Obtain region of interest (hand drawn boxes) from radiograph than scale them to have the same size
    '''
    smallImages = np.zeros((number_samples, rewidth * reheight))
    #radiographs = [preprocess_radiograph(radiograph) for radiograph in radiographs]
    for ind, radiograph in enumerate(radiographs):
        this_box = four_incisor_bbox[ind] # Get the hand drawn box for current radiograph
        cutImage = radiograph[this_box[1]:this_box[3], this_box[0]:this_box[2]]
        result = cv2.resize(cutImage, (rewidth, reheight), interpolation=cv2.INTER_NEAREST)
        smallImages[ind] = result.flatten()
    #smallImages = np.asarray(smallImages)
    return smallImages
         
def best_seg(mean, evecs, image, is_upper, largest_boxes, width, height, show=False):
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
    #print(largest_boxes)

    if is_upper:
        b1 = int(w/2 - w/10)
        b2 = int(w/2 + w/10)
        a1 = int(largest_boxes[1]) 
        a2 = int(largest_boxes[3])
    else:
        b1 = int(w/2 - w/12)
        b2 = int(w/2 + w/12)
        a1 = int(largest_boxes[5])
        a2 = int(largest_boxes[7])
    search_region = [(b1, a1), (b2, a2)]

    best_score = float("inf")
    best_score_bbox = [(-1, -1), (-1, -1)]
    best_score_img = np.zeros((width, height))
    for wscale in np.arange(0.7, 1.3, 0.1): # start 0.7 stop 1.3 step 0.1 -- Try different scales for width
        for hscale in np.arange(0.7, 1.3, 0.1): # start 0.7 stop 1.3 step 0.1 -- Try different scales for hight
            slideW = int(width * wscale)
            slideH = int(height * hscale)
            #print('winw winH image',winW,winH)
            for (x, y, window) in slide(image, search_region, step=36, window = (slideW, slideH)):
                # If the size of the window is not what we wanted it means that we are sliding out of the image
                # so we go to the next window
                if window.shape[0] != slideH or window.shape[1] != slideW: continue
                # Resize window to unit size 
                reCut = cv2.resize(window, (width, height))
                X = reCut.flatten()
                #Project X on the space spanned by the vectors in evecs. Mean is the average image.
                Y = project(evecs, X, mean)
                #Reconstruct an image based on its PCA-coefficients Y, the evecs and the average mean.
                Xacc = reconstruct(evecs, Y, mean)
                # Calculate score based on the error to reconstruct the image
                score = np.linalg.norm(Xacc - X)
                if score < best_score:
                    best_score = score
                    best_score_bbox = [x,y,x + slideW,y + slideH]
                    best_score_img = reCut
        if show:
            cv2.imshow('IF recut', best_score_img)
    return (best_score_bbox)

def pca(X, nb_components):
    mu = np.average(X, axis=0)
    X = np.add(X, -mu.transpose(), out=X, casting="unsafe")

    eigenvalues, eigenvectors = np.linalg.eig(np.dot(X, np.transpose(X)))
    eigenvectors = np.dot(np.transpose(X), eigenvectors)
    eig = zip(eigenvalues, np.transpose(eigenvectors))
    eig = map(lambda x: (x[0] * np.linalg.norm(x[1]),
                         x[1] / np.linalg.norm(x[1])), eig)
    eig = sorted(eig, reverse=True, key=lambda x: abs(x[0]))
    eig = eig[:nb_components]

    eigenvalues, eigenvectors = map(np.array, zip(*eig))

    return np.transpose(eigenvectors), mu



def estimate(rad, isupper, preprocessed_r, index, number_samples, coord, allcoord, show, b_model):
    ''' 
    rad - radiograph we want to find the best box on
    isupper - is it upper or lower incisiors
    preprocessed_r - all the radiographs
    coord - coordonates of the largest drawn box
    allcoord - coordonates of all upper/lower incisors
    show - print box if True
    b_model - build model if True, load model else
    '''
    if isupper:
        width = coord[2] - coord[0]
        height = coord[3] - coord[1] 
        if b_model:
            data = cut_radiographs(np.asarray(preprocessed_r[0:number_samples]), isupper, allcoord, width, height, number_samples)
            [eigen_vec, mean] = pca(data, 10)
            
            filename = 'eigen_vec_upper.sav'
            pickle.dump(eigen_vec, open(filename, 'wb'))
    
            filename = 'mean_upper.sav'
            pickle.dump(mean, open(filename, 'wb'))        
            
            print('  Outputed to file')
        else: 
            eigen_vec = pickle.load(open('eigen_vec_upper.sav', 'rb'))
            mean = pickle.load(open('mean_upper.sav', 'rb'))  
    else:
        width = coord[6]-coord[4]
        height = coord[7]-coord[5]
        if b_model:
            data = cut_radiographs(np.asarray(preprocessed_r[0:number_samples]), isupper, allcoord, width, height, number_samples)
            [eigen_vec, mean] = pca(data, 10)
            filename = 'eigen_vec_lower.sav'
            pickle.dump(eigen_vec, open(filename, 'wb'))
    
            filename = 'mean_lower.sav'
            pickle.dump(mean, open(filename, 'wb'))        
            
            print('  Outputed to file')
        else: 
            eigen_vec = pickle.load(open('eigen_vec_lower.sav', 'rb'))
            mean = pickle.load(open('mean_lower.sav', 'rb')) 

    # Find the region of the radiograph that matches best with the appearance model
    best_coord = best_seg(mean, eigen_vec, rad, isupper, coord, width, height, False)

    if show:
        img = rad.copy()
        cv2.rectangle(img, (int(best_coord[0]), int(best_coord[1])), (int(best_coord[2]), int(best_coord[3])), (255, 0, 0), 5)
        cv2.imshow('Radiograph with best box', img)
        directory = '../Plot/Boxes/box' + str("%02d" % (index))+'.png'
        dir_path = os.path.join(os.getcwd(), directory)
        cv2.imwrite(dir_path, img)
        cv2.waitKey(0)
    return best_coord
