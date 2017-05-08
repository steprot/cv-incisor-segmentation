# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, render_landmark_over_image, translate_to_origin, compute_centroids, scale_to_unit, tooth_from_vector_to_matrix, align_teeth_to_mean_shape, tooth_from_matrix_to_vector

'''
    Prints teeth's landmark points over the radiographs.
    Parameters: 
         teeth_landmarks; 3D table containing the 80 landmark points per tooth, per sample.
'''
def print_landmarks_over_radiographs(teeth_landmarks):
    i = 0
    while i < number_samples:
        j = 0
        directory = '../_Data/Radiographs/' + str("%02d" % (i+1)) + '.tif'
        dir_path = os.path.join(os.getcwd(), directory)
        img = cv2.imread(dir_path)
        while j < number_teeth:
            # Call the helper function
            img = render_landmark_over_image(img, teeth_landmarks[j,i,:])
            # Go to the next tooth
            j += 1
        # Got the next sample   
        i += 1

'''
    Prints the single tooth landmark points over a black image. 
    Parameter: 
        tooth_landmarks; array containing the 80 landmark points of the tooth. 
'''
def print_landmarks(tooth_landmarks):
    
    # Transform the array in a matrix 
    i = 0
    points = []
    while i < len(tooth_landmarks)-1: # -1 because then I add +1
        points.append([float(tooth_landmarks[i]), float(tooth_landmarks[i+1])])
        i += 2 # To go to the next couple
    points = np.array(points)

    max_y = points[:, 0].max()
    min_y = points[:, 0].min()
    max_x = points[:, 1].max()
    min_x = points[:, 1].min()
    
    img = np.zeros((int((max_y - min_y) * 1.1), int((max_x - min_x) * 1.1)))
    
    for i in range(len(points)):
        img[int(points[i, 0] - min_y), int(points[i, 1] - min_x)] = 1
    
    cv2.imshow('Rendered shape', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    Main function of the project.
'''
if __name__ == '__main__':
    
    # *** Read the landmarks ****
    number_teeth = 8 
    number_samples = 3
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
            # print the landmark
            # print(l)
            landmark.append(np.array(l))

            # Got the next sample
            j +=1
            
        teeth_landmarks.append(np.array(landmark))
        # Go to the next tooth
        i +=1
        
    teeth_landmarks = np.array(teeth_landmarks)  
        
    # *** Print the teeth_landmarks over  *** 
    #print_landmarks_over_radiographs(teeth_landmarks)
    
    # *** Compute the centroids from the landmarks ***
    
    """ teeth_landmarks[i] RETURNS ALL THE TEETH FROM ONE sample ! """
    
    centroids = [] # Contains the centers of the means, not the individual tooth
    
    for i in range(0,number_teeth):
        #centroids.append(compute_centroids(teeth_landmarks[i]))
        centroids.append(compute_centroids(teeth_landmarks[i]))
    #print('Len of centroids', len(centroids)) # it is 8
    
    # *** Compute the mean_shape from the landmarks ***
    mean_shape = []
    for i in range(0,number_teeth):
        mean_shape.append(np.mean(teeth_landmarks[i], axis = 0))
    #print('Len of mean_shape', len(mean_shape)) # it is 8
    # print(mean_shape)
    
    # *** Translate the landmarks around the origin ***
    around_origin = []
    for i in range(0,number_teeth): 
        for j in range(0,number_samples):
            orig = translate_to_origin(teeth_landmarks[i,j])
            around_origin.append(orig)
    #print('Len of around_origin', len(around_origin)) # it is n*8
    # print(around_origin)

    # *** Scale the landmarks to have the norm of the shape equal to 1 *** 
    
    # This is the scaled shape of the teeth, centred in the origin
    scaled_shape_from_means = []
    for i in range(0,number_teeth):
        scaled_shape_from_means.append(scale_to_unit(tooth_from_vector_to_matrix(mean_shape[i]), centroids[i]))
    # print(scaled_shape_from_means)
    
    # Flag to break the while loop when the model converges
    converge = False
    #pa_result = np.zeros_like(around_origin)# Generalized Procrustes Analysis results
    pa_result = []
    while not converge:
        for index, element in enumerate(around_origin):
            # For each tooth align_teeth_to_mean_shape takes 1 row (n*80) and the scaled_mean for the same tooth
            #around_origin[index] = align_teeth_to_mean_shape(tooth_from_matrix_to_vector(around_origin[index]), tooth_from_matrix_to_vector(scaled_shape_from_means[index]))
            
            """I get different results if I do the two things below, why?"""
            #pa_result[index] = align_teeth_to_mean_shape(tooth_from_matrix_to_vector(around_origin[index]), tooth_from_matrix_to_vector(scaled_shape_from_means[index]))
            pa_result.append(align_teeth_to_mean_shape(tooth_from_matrix_to_vector(around_origin[index]), tooth_from_matrix_to_vector(scaled_shape_from_means[index])))
            #print(around_origin[index] )
            break
        break     
   
    # Do PCA
     
    # Covariance matrix  
    """ Error when creating covariance matrix"""
    #TODO -- Needs to be corrected
    print(pa_result)
    pa_result_matrix = []
    for i in range(len(pa_result)):
        pa_result_matrix.append(tooth_from_vector_to_matrix(pa_result[i]))
    print(pa_result_matrix)
    C = np.cov(pa_result_matrix, rowvar=0)
    
    eigvals, eigvecs = np.linalg.eigh(C) # Get eigenvalues and eigenvectors
    indeces = np.argsort(-eigvals)   # Sort them in descending order
    eigvals = eigvals[indeces]
    eigvecs = eigvecs[:, indeces]
    
    scores = np.dot(pa_result_matrix, eigvecs)
    mean_scores = np.dot(mean_shape, eigvecs)
    variance = np.cumsum(eigvals/np.sum(eigvals))
    
    print scores, mean_scores, variance
        
        
        