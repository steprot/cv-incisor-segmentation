# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, render_landmark_over_image, translate_to_origin, scale_to_unit, tooth_from_vector_to_matrix, align_teeth_to_mean_shape, tooth_from_matrix_to_vector, compute_new_mean_shape, get_tooth_centroid, plot_procrustes

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
    
    # ***** Read the landmarks ******
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
            # print(l)
            landmark.append(np.array(l))

            # Got the next sample
            j +=1
        
        teeth_landmarks.append(np.array(landmark))
        # Go to the next tooth
        i +=1
        
    teeth_landmarks = np.array(teeth_landmarks) # teeth_landmarks.shape is 8*n*80
        
    # ***** Print the teeth_landmarks over radiographs ***** 
    # print_landmarks_over_radiographs(teeth_landmarks)

      

    
    
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
        # print_landmarks(np.array(np.mean(around_origin[i], axis = 0)))
        mean_shape.append(np.array(np.mean(around_origin_scaled[i], axis = 0))) # len of mean_shape is 8
    mean_shape = np.array(mean_shape)
    # print('mean_shape shape', mean_shape.shape) # 8x80
    # print('mean_shape', mean_shape)
    
    
    # ***** Compute the mean_centroids from the landmarks *****
    mean_centroids = [] # Contains the centroids of the mean shape
    for i in range(0, number_teeth):
        mean_centroids.append(get_tooth_centroid(mean_shape[i]))
    mean_centroids = np.array(mean_centroids)
    
    ## ***** Scale the mean landmarks to have the norm of the shape equal to 1 ***** 
    #scaled_shape_from_means = []
    #for i in range(number_teeth):
    #    scaled_shape_from_means.append(np.array(scale_to_unit(tooth_from_vector_to_matrix(mean_shape[i]), 
    #                                                          mean_centroids[i]))) # len of scaled_shape_from_means is 8
    #scaled_shape_from_means = np.array(scaled_shape_from_means)
    ## Transform the scaled_shape_from_means from 8x40x2 in 8x80
    #tmp = []
    #for i in range(number_teeth):
    #    tmp.append(tooth_from_matrix_to_vector(scaled_shape_from_means[i]))
    #scaled_shape_from_means = np.array(tmp)
    ## print('scaled_shape_from_means shape', scaled_shape_from_means.shape)
    ## print('scaled_shape_from_means', scaled_shape_from_means)
    
    # ***** Do the Generalized Procrustes Analysis on the landmarks ***** 
    
    while True:
        
        print()
        #Alligne shapes
        aligned_shape = np.copy(around_origin_scaled) # we don't need the copy, we will update it in the for loops with the aligned shapes
        for i in range(0, number_teeth):
            for j in range(0, number_samples):
                aligned_shape[i,j] = align_teeth_to_mean_shape(around_origin_scaled[i,j], mean_shape[i] )
        
        #Calculate new mean
        new_mean_shape = [] # Matrix of nr of teeth x 80
        for i in range(0, number_teeth):
        # print_landmarks(np.array(np.mean(around_origin[i], axis = 0)))
            new_mean_shape.append(np.array(np.mean(aligned_shape[i], axis = 0))) # len of mean_shape is 8
        new_mean_shape = np.array(new_mean_shape)
        
        
        
        # Scaleing and translating new mean shapes
        for i in range(0, number_teeth):
            new_mean_shape[i] = scale_to_unit(new_mean_shape[i], get_tooth_centroid((new_mean_shape[i])))
            new_mean_shape[i] = tooth_from_matrix_to_vector(translate_to_origin(new_mean_shape[i]))
        
        if (mean_shape - new_mean_shape < 1e-10).all():
            print('Yeyy it works!')
            break
        
        mean_shape = new_mean_shape
        print('Still looping :/ ')
        
    
    
#    # ***** Do the Generalized Procrustes Analysis on the landmarks ***** 
#    converge = False # Flag to break the while loop when the model converges
#    pa_result = [] # Procrustes Analysis result
#    ''' this is just the first round, to create the pa_result list. then in the while loop we can update it, there are already the indexes '''
#    for i in range(number_teeth):
#        pa_result_row = []
#        for j in range(number_samples):
#            pa_result_row.append(np.array(align_teeth_to_mean_shape(around_origin_scaled[i,j], 
#                                            tooth_from_matrix_to_vector(scaled_shape_from_means[i]))))
#        pa_result.append(np.array(pa_result_row))
#    pa_result = np.array(pa_result)
#    # print('pa_result shape', pa_result.shape) 8xnx80
#                                                                    
#    # Till there is not convergence;
#    while not converge:
#        # scaled_shape_from_means and new_scaled_shape_from_means they have the same shape
#        # Compute the new_mean from the values just obtained. 
#        new_scaled_shape_from_means = compute_new_mean_shape(pa_result, number_teeth, number_samples)
#    
#        print('Diffeneces: ', scaled_shape_from_means - new_scaled_shape_from_means)
#        if (scaled_shape_from_means - new_scaled_shape_from_means < 1e-3).all():
#            converge = True
#
#            print('fine ciclo while')
#            break
            
    # print('pa_result shape', pa_result.shape)
    
    # plot_procrustes(scaled_shape_from_means[0], pa_result[0], incisor_nr=0, save=False)
      
    # ***** Do PCA *****
    # Covariance matrix  
    """ Error when creating covariance matrix"""
    #TODO -- Needs to be corrected
    
    #pa_result_matrix = []
    #for i in range(len(pa_result)):
    #    pa_result_matrix.append(tooth_from_vector_to_matrix(pa_result[i]))
        
    #pa_result_matrix = np.zeros((number_teeth, number_samples, 80))
    #for tooth_ind in range(number_teeth):
    #    for sample_ind in range(number_samples):
    #        pa_result_matrix[tooth_ind, sample_ind] = pa_result[sample_ind*number_teeth + tooth_ind]
    #   
    ## Create the covariance matrix  
    #C = np.cov(pa_result_matrix, rowvar=0)
    #
    #eigvals, eigvecs = np.linalg.eigh(C) # Get eigenvalues and eigenvectors
    #indeces = np.argsort(-eigvals)   # Sort them in descending order
    #eigvals = eigvals[indeces]
    #eigvecs = eigvecs[:, indeces]
    #
    #scores = np.dot(pa_result_matrix, eigvecs)
    #mean_scores = np.dot(mean_shape, eigvecs)
    #variance = np.cumsum(eigvals/np.sum(eigvals))
    #
    #print scores, mean_scores, variance
    #    
        
        