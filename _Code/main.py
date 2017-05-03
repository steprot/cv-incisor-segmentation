# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, render_landmark, render_landmark_over_image

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
            #render_landmark(landmark)
            tooth_landmarks[j-1,:] = landmark.as_vector()

            # Got the next sample
            j +=1
            
        teeth_landmarks[i-1] = np.hstack(tooth_landmarks)
        # Go to the next tooth
        i +=1
        
    #Print the landmarks over the radiographs **********************************
    
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