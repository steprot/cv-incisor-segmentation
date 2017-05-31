# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, translate_to_origin, scale_to_unit, tooth_from_vector_to_matrix, align_teeth_to_mean_shape, tooth_from_matrix_to_vector, get_tooth_centroid
from sklearn.decomposition import PCA
import preprocessing as pp
import boxes as bx
import fitter as fit
from visualise import render_landmarks, print_landmarks_over_radiographs,plot_procrustes, render_landmark_over_image, __get_colors, render_model_over_image
from estimate import estimate

def getcut(img):
    '''
    This is a bad idea, get the parameters from the hand draw shapes
    '''
    h, w = img.shape
    #print('img.shape', img.shape)
    b1 = int(w/2 - w/10)
    b2 = int(w/2 + w/10)
    a1 = int(h/2 + h/7) - 350
    a2 = int(h/2 + h/6) - 40
    
    crp = img[a1 :a2, b1:b2]
    #crp = img[b1:b2, a1 :a2]
    cv2.imshow("Cropped image",crp)

'''
    Main function of the project.
'''
if __name__ == '__main__':
    
    # Global variables 
    n_components = 8
    draw_handboxes = False 
    
    # ***** Read the landmarks ******
    number_teeth = 8 
    number_samples = 14
    teeth_landmarks = [] # 3D array: 8*number_samples*80
        
    i = 1
    while i <= number_teeth: # For all the different teeth 
        j = 1 
        tooth_landmarks = np.zeros((number_samples, 80))
        landmark = []
        while j <= number_samples: # For the same tooth but for the different persons
            
            # Specify the name where to read the landmarks from
            directory = '../_Data/Landmarks/original/landmarks' + str(j) + '-' + str(i) + '.txt'
            dir_path = os.path.join(os.getcwd(), directory)
            
            l = Landmarks(dir_path).as_vector()
            # print(l)
            landmark.append(np.array(l))

            # Got the next sample
            j +=1
        teeth_landmarks.append(np.array(landmark))
        # Go to the next tooth
        i +=1
        
    teeth_landmarks = np.array(teeth_landmarks) # teeth_landmarks.shape is 8*n*80
    print('* Landmarks loaded *')
     
     
    # ***** Print the teeth_landmarks over radiographs ***** 
    #print_landmarks_over_radiographs(teeth_landmarks)
    
    # ***** Scale landmarks ***** 
    scaled = []
    for i in range(number_teeth):
        scaled_row = []
        for j in range(number_samples):
            scaled_row.append(np.array(tooth_from_matrix_to_vector(scale_to_unit(tooth_from_vector_to_matrix(teeth_landmarks[i,j]), get_tooth_centroid(teeth_landmarks[i,j])))))
        scaled.append(np.array(scaled_row))
    scaled =  np.array(scaled)
    
    
    # ***** Translate the scaled lendmarks around the origin *****
    around_origin_scaled = []
    for i in range(0, number_teeth): 
        around_origin_row = []
        for j in range(0, number_samples):
            around_origin_row.append(tooth_from_matrix_to_vector(np.array(translate_to_origin(scaled[i,j]))))
        around_origin_scaled.append(np.array(around_origin_row))
    around_origin_scaled = np.array(around_origin_scaled) # shape of around_origin is 8xnx80
    
    
    # ***** Compute the mean_shape from the around origin scaled landmarks *****   
    mean_shape = [] # Matrix of nr of teeth x 80
    for i in range(0, number_teeth):
        mean_shape.append(np.array(np.mean(around_origin_scaled[i], axis = 0))) # len of mean_shape is 8
    mean_shape = np.array(mean_shape)
    
    
    # ***** Compute the mean_centroids from the landmarks *****
    mean_centroids = [] # Contains the centroids of the mean shape
    for i in range(0, number_teeth):
        mean_centroids.append(get_tooth_centroid(mean_shape[i]))
    mean_centroids = np.array(mean_centroids)
    
    print('* Starting Procrustes Analysis *')
    
    
    # ***** Do the Generalized Procrustes Analysis on the landmarks ***** 
    while True:

        #Align shapes
        aligned_shape = np.copy(around_origin_scaled) # we don't need the copy, we will update it in the for loops with the aligned shapes
        for i in range(0, number_teeth):
            for j in range(0, number_samples):
                aligned_shape[i,j] = align_teeth_to_mean_shape(around_origin_scaled[i,j], mean_shape[i] )
        
        #Calculate new mean
        new_mean_shape = [] # Matrix of nr_teeth x 80
        for i in range(0, number_teeth):
            new_mean_shape.append(np.array(np.mean(aligned_shape[i], axis = 0))) # len of mean_shape is 8
        new_mean_shape = np.array(new_mean_shape)
        
        # Scaling and translating new mean shapes
        for i in range(0, number_teeth):
            new_mean_shape[i] = scale_to_unit(new_mean_shape[i], get_tooth_centroid((new_mean_shape[i])))
            new_mean_shape[i] = tooth_from_matrix_to_vector(translate_to_origin(new_mean_shape[i]))
        
        # Terminating condition
        if (mean_shape - new_mean_shape < 1e-10).all():
            break
        
        mean_shape = new_mean_shape
    
    
    # ***** Do PCA *****
    print('* Starting PCA *')
    
    # Covariance matrix  
    reduced_dim = []
    for i in range(len(aligned_shape)):
        
        #pca_res = PCA(.99) 
        pca_res = PCA(n_components) 
                           # Instead of setting th nr of components, we make sure 
                           # that we capture 99% of the variance 
                           # this gives eigenvectors with different length! for each incisor so 
                           # it was better to set the nr to a fixed value, 8 covers around 99%
        # From pca_res we can obtain :
        # components_
        # explained_variance_
        # explained_variance_ratio_
        # components_
        # mean_
        # n_components_
        pca_res.fit(aligned_shape[i])
        reduced_dim.append(pca_res.transform(aligned_shape[i])) # Reduce dimensionality of the training data

    reduced_dim = np.asarray(reduced_dim)
    
    
    # ***** Print the ASM of the landmarks *****
    """ Be careful when printing, the order is  k[i,1],k[i,0] not the other way around!"""
    #for i in range(number_teeth):
    #    plot_procrustes(mean_shape[i],aligned_shape[i], i+1, False)       
    
    
    # ***** Do pre-processing of the images *****
    print('* Starting preprocessing *')
    # radiographs contains the raw radiographs images
    radiographs = pp.load_radiographs(number_samples,False)
    # Preprocess the images
    preprocessed_r = []
    for i in range(len(radiographs)):
        preprocessed_r.append(pp.preprocess_radiograph(radiographs[i]))
        #cv2.imshow("preprocessed in main", preprocessed_r[i])
        #cv2.waitKey(0)
    # Find the edges of the radiographs
    edges = []
    for i in range(len(preprocessed_r)):
        # Finding the edges 
        edges.append(pp.togradient_sobel(preprocessed_r[i]))
        #cv2.imshow('Edges', edges[i])
        #cv2.waitKey(0)
    print('* Preprocessing finished *')

    # ***** Ask the user to draw the boxes around the jaws ***** 
    if draw_handboxes: 
        teeth_boxes = []
        for i in range(number_samples): # number_samples
            teeth_boxes_row = []
            teeth_boxes_row.append(bx.get_box_per_jaw(radiographs[i], i+1, 'upper'))
            teeth_boxes_row.append(bx.get_box_per_jaw(radiographs[i], i+1, 'lower'))    
            teeth_boxes_row = np.asarray(teeth_boxes_row)
            teeth_boxes_row = np.hstack(teeth_boxes_row)
            # Print the detailed boxes on the image 
            # bx.print_boxes_on_tooth(teeth_boxes_row, radiographs[i])
            teeth_boxes.append(teeth_boxes_row)
        
        teeth_boxes = np.asarray(teeth_boxes) # dimension is n_samples*8
        # If it is asked for the new boxes, then save them 
        bx.save_boxes(teeth_boxes)
        print('* Boxes saved *')

    #mean_box = bx.get_mean_boxes(teeth_boxes) # dimension is 1x8
    
    
    # ***** Reading the boxes from file and get the largest one *****
    # Getting the largest boxes from the file     
    boxes_from_file = bx.read_boxes_from_file()   
    largest_b = bx.get_largest_boxes(boxes_from_file)
    upper = []
    lower = []
    for i in range(len(boxes_from_file)):
        upper.append([boxes_from_file[i][0], boxes_from_file[i][1], boxes_from_file[i][2], boxes_from_file[i][3] ])
        lower.append([boxes_from_file[i][4], boxes_from_file[i][5], boxes_from_file[i][6], boxes_from_file[i][7] ])
    print('* Largest box obtained *')
    
    #for i in range(number_samples):
    #    bx.print_boxes_on_teeth(largest_b, radiographs[i])
    
    
    # ***** Sharp the boxes ***** 
    """  
    IMPORTANT NOTES REGARDING ESTIMATE
    - you have to spcify which tooth are you looking for
    - you have to spacify if it is upper or lower (as last parameter input upper/lower) -> this can be made dynamic
    """
    toothnr = 0
    #rad_nr = 9
    estimates = []
    for rad_nr in range(14):
        e = estimate(rad_nr, mean_shape[toothnr], toothnr, preprocessed_r, largest_b, upper, True)
	estimates.append(e)
    
    print('* Found boxes for the upper teeth *')
    print('Estimates upper:', estimates)
    
    toothnr = 6
    for rad_nr in range(14):
        estimates[rad_nr].extend(estimate(rad_nr, mean_shape[toothnr], toothnr, preprocessed_r, largest_b, lower, False))
	
    print('* Found boxes for the lower teeth *')
    print('Estimates ALLLLLL:', estimates)
    
    print('estimates.len', len(estimates))
    print('largest_b.len', len(largest_b))
    
    
    # ***** Apply and fit the model over the image ***** 
    
    # Detailed boxes is 8*4, and must be computed for each sample 
    detailed_boxes = fit.get_detailed_boxes(largest_b)
    
    # Get colours to print the teeth over the imgages 
    colors = __get_colors(number_teeth)
    new_points = []
    # Loop over every image 
    for j in range(number_samples):
        # Loop over every tooth 
        for i in range(number_teeth):
            # Apply the model over the boxes 
            img, newpoints = fit.fit_asm_model_to_box(mean_shape[i], detailed_boxes[i], radiographs[j], 1000, colors[i], edges[j])
            # Smooth the obtained points 
            newpoints = fit.smooth_model(newpoints)
            render_model_over_image(newpoints, radiographs[j], i+1, colors[i], True)
            new_points.append(newpoints)
    