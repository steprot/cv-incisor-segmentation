# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
from landmark import Landmarks, render_landmark, render_landmark_over_image, translate_to_origin, compute_mean

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
    number_samples = 2
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
    #print(np.array(teeth_landmarks[0]).shape)  
    
    # *** Print the teeth_landmarks over  *** 
    print_landmarks_over_radiographs(teeth_landmarks)
    
    # *** Compute the mean_teeth from the landmarks ****
    mean_teeth = []
    for i in range(0,number_teeth):
        mean_teeth.append(compute_mean(teeth_landmarks[i]))
    
    print(mean_teeth)  
    
    ##print(tooth_landamarks)
    ##       
    #aroundOrigin = []
    #for i in range(0,number_teeth): 
    #    for j in range(0,number_samples):
    #        orig  = translate_to_origin(teeth_landmarks[i,j])
    #        aroundOrigin.append(orig)
    #print(aroundOrigin)
    #print 'End'
    #
    #