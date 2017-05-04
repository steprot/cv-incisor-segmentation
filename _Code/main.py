# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, render_landmark, render_landmark_over_image, translate_to_origin, compute_centroid
from activeshapemodel import ActiveShapeModel

# Print the landmarks over the radiographs *********************************
def print_landmarks_over_radiographs(teeth_landmarks):
    i = 0
    while i < number_samples:
        j = 0
        directory = '../_Data/Radiographs/0' + str("%01d" % (i+1)) + '.tif'
        dir_path = os.path.join(os.getcwd(), directory)
        img = cv2.imread(dir_path)
        while j < number_teeth:
            tooth_landamarks = teeth_landmarks[j,:]
            tooth_landamarks = tooth_landamarks[i*80:i*80+80]
            img = render_landmark_over_image(img, tooth_landamarks)
            
            # Go to the next tooth
            j += 1
        # Got the next sample   
        i += 1

if __name__ == '__main__':
    
    # Read the landmarks *******************************************************

    number_teeth = 8
    number_samples = 2
    teeth_landmarks = np.zeros((number_teeth, number_samples * 80))
        
    i = 1
    while i <= number_teeth: # For all the different teeth 
        j = 1 
        tooth_landmarks = np.zeros((number_samples, 80))
        while j <= number_samples: # For the same tooth but for the different persons
            
            # Specify the name where to read the landmarks from
            directory = '../_Data/Landmarks/original/landmarks' + str(j) + '-' + str(i) + '.txt'
            dir_path = os.path.join(os.getcwd(), directory)
            
            landmark = Landmarks(dir_path)
            # Print it 
            #render_landmark(landmark.as_vector())
            tooth_landmarks[j-1,:] = landmark.as_vector()

            # Got the next sample
            j +=1
            
        teeth_landmarks[i-1] = np.hstack(tooth_landmarks)
        # Go to the next tooth
        i +=1
        
    # Print the teeth over the radiographs
    # print_landmarks_over_radiographs(teeth_landmarks)
                
    # Fit the ActiveShapeModel and compute the mean sample for each tooth ******
    
    mean_teeth_landmarks = np.zeros((number_teeth, 80))
    
    i = 0 
    while i < number_teeth:
        activeShapeModel = ActiveShapeModel(teeth_landmarks[i,:], number_samples)
        mean_teeth_landmarks[i] = activeShapeModel.compute_mean()
        
        # Print it 
        # render_landmark(mean_teeth_landmarks[i])
        # print(mean_teeth_landmarks)
        i += 1
    
    # Translate the landmarks to the origin ************************************
    
    i = 0
    teeth_landmarks_translated_to_origin = np.zeros((number_teeth, number_samples * 80))
    while i < number_samples:
        j = 0
        while j < number_teeth:
            tooth_landamarks = teeth_landmarks[j,:]
            tooth_landamarks = tooth_landamarks[i*80:i*80+80]
            
            origin_landmarks = translate_to_origin(tooth_landamarks, None)
            
            #teeth_landmarks_translated_to_origin[j, i*80:i*80+80] = compute_centroid(tooth_landamarks, None)

            # Go to the next tooth
            j += 1
        # Got the next sample   
        i += 1
        
    mean = compute_centroid(origin_landmarks, None)
    print(mean)
        
    # Print the teeth over the radiographs
    # print_landmarks_over_radiographs(teeth_landmarks_translated_to_origin)
        
        
        
    