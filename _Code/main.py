# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, translate_to_origin, scale_to_unit, tooth_from_vector_to_matrix, align_teeth_to_mean_shape, tooth_from_matrix_to_vector, get_tooth_centroid
from sklearn.decomposition import PCA
from preprocessing import load_radiographs, preprocess_radiograph
from visualise import render_landmarks, print_landmarks_over_radiographs,plot_procrustes
from estimate import estimate
import init 

def getcut(img):
    """ this is a bad isea, get the parameters from the hand draw shapes """
    h, w = img.shape
    print(img.shape)
    b1 = int(w/2 - w/10)
    b2 = int(w/2 + w/10)
    a1 = int(h/2 + h/7) - 350
    a2 = int(h/2 + h/6) - 40
    
    crp = img[a1 :a2, b1:b2]
    #crp = img[b1:b2, a1 :a2]
    cv2.imshow("cut it damn",crp)

'''
    Main function of the project.
'''
if __name__ == '__main__':
    
    # ***** Read the landmarks ******
    number_teeth = 8 
    number_samples = 14
    teeth_landmarks = [] # 3D array: 8*number_samples*80
    n_components = 8
        
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
    
    
    # ***** Do the Generalized Procrustes Analysis on the landmarks ***** 
    while True:

        #Alligne shapes
        aligned_shape = np.copy(around_origin_scaled) # we don't need the copy, we will update it in the for loops with the aligned shapes
        for i in range(0, number_teeth):
            for j in range(0, number_samples):
                aligned_shape[i,j] = align_teeth_to_mean_shape(around_origin_scaled[i,j], mean_shape[i] )
        
        #Calculate new mean
        new_mean_shape = [] # Matrix of nr of teeth x 80
        for i in range(0, number_teeth):
            new_mean_shape.append(np.array(np.mean(aligned_shape[i], axis = 0))) # len of mean_shape is 8
        new_mean_shape = np.array(new_mean_shape)
        
        # Scaling and translating new mean shapes
        for i in range(0, number_teeth):
            new_mean_shape[i] = scale_to_unit(new_mean_shape[i], get_tooth_centroid((new_mean_shape[i])))
            new_mean_shape[i] = tooth_from_matrix_to_vector(translate_to_origin(new_mean_shape[i]))
        
        if (mean_shape - new_mean_shape < 1e-10).all():
            break
        
        mean_shape = new_mean_shape
    
    # ***** Do PCA *****
    
    # Covariance matrix  
    reduced_dim = []
    for i in range(len(aligned_shape)):
        
        #pca_res = PCA(.99) 
        pca_res = PCA(n_components) 
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
        pca_res.fit(aligned_shape[i])

        reduced_dim.append(pca_res.transform(aligned_shape[i])) # Reduce dimensionality of the training data

    reduced_dim = np.asarray(reduced_dim)
    
    #plot_procrustes(mean_shape[0],aligned_shape[0],0,False )
    
    #render_landmarks(aligned_shape[0])         
    
    # ***** Do pre-processing of the images *****
    # radiographs contains the raw radiographs images
    radiographs = load_radiographs(number_samples,False)
    preprocessed_r = []
    for i in range(len(radiographs)):
        preprocessed_r.append(preprocess_radiograph(radiographs[i]))
        
    #estimate(mean_shape[0],1,np.asarray(preprocessed_r))
    #getcut(preprocessed_r[0])
    
    # ***** Ask the user to draw the boxes around the jaws ***** 
    
    teeth_boxes = []
    for i in range(2): # number_samples
        teeth_boxes_row = []
        teeth_boxes_row.append(init.get_box_per_jaw(radiographs[i], i, 'upper'))
        teeth_boxes_row.append(init.get_box_per_jaw(radiographs[i], i, 'lower'))    
        teeth_boxes_row = np.asarray(teeth_boxes_row)
        teeth_boxes_row = np.hstack(teeth_boxes_row)
        teeth_boxes.append(teeth_boxes_row)
    teeth_boxes = np.asarray(teeth_boxes)
    #print(teeth_boxes)
    #print(teeth_boxes_row.shape)
            
