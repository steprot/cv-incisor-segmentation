# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch
import landmark as lm
from sklearn.decomposition import PCA
import preprocessing as pp
import boxes as bx
import fitter as fit
import visualise as visual
from estimate import estimate

'''
    Main function of the project.
'''

def load_landmarks(number_teeth,number_samples, directory):
    i = 1
    teeth_landmarks = [] # 3D array: 8*number_samples*80
    path ='../_Data/Landmarks/' + str(directory)
    while i <= number_teeth: # For all the different teeth 
        j = 1 
        landmark = []
        while j <= number_samples: # For the same tooth but for the different persons
            # Specify the name where to read the landmarks from
            directory = path + str(j) + '-' + str(i) + '.txt'
            dir_path = os.path.join(os.getcwd(), directory)
            
            l = lm.Landmarks(dir_path).as_vector()
            # print(l)
            landmark.append(np.array(l))

            # Got the next sample
            j +=1
        teeth_landmarks.append(np.array(landmark))
        # Go to the next tooth
        i +=1
        
    teeth_landmarks = np.array(teeth_landmarks) # teeth_landmarks.shape is 8*n*80
    return teeth_landmarks
    
    
def pre_procrustes(number_teeth,number_samples,teeth_landmarks):
        # ***** Scale landmarks ***** 
    scaled = []
    for i in range(number_teeth):
        scaled_row = []
        for j in range(number_samples):
            scaled_row.append(np.array(lm.tooth_from_matrix_to_vector(lm.scale_to_unit(lm.tooth_from_vector_to_matrix(teeth_landmarks[i,j]), \
                                                                      lm.get_tooth_centroid(teeth_landmarks[i,j])))))
        scaled.append(np.array(scaled_row))
    scaled =  np.array(scaled)
    
    
    # ***** Translate the scaled lendmarks around the origin *****
    around_origin_scaled = []
    for i in range(0, number_teeth): 
        around_origin_row = []
        for j in range(0, number_samples):
            around_origin_row.append(lm.tooth_from_matrix_to_vector(np.array(lm.translate_to_origin(scaled[i,j]))))
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
        mean_centroids.append(lm.get_tooth_centroid(mean_shape[i]))
    mean_centroids = np.array(mean_centroids)    
    
    return around_origin_scaled, mean_shape
    
    
def procrustes(around_origin_scaled,number_teeth, mean_shape):
    while True:
        #Align shapes
        aligned_shape = np.copy(around_origin_scaled) # we don't need the copy, we will update it in the for loops with the aligned shapes
        for i in range(0, number_teeth):
            for j in range(0, number_samples):
                aligned_shape[i,j] = lm.align_teeth_to_mean_shape(around_origin_scaled[i,j], mean_shape[i] )
        
        #Calculate new mean
        new_mean_shape = [] # Matrix of nr_teeth x 80
        for i in range(0, number_teeth):
            new_mean_shape.append(np.array(np.mean(aligned_shape[i], axis = 0))) # len of mean_shape is 8
        new_mean_shape = np.array(new_mean_shape)
        
        # Scaling and translating new mean shapes
        for i in range(0, number_teeth):
            new_mean_shape[i] = lm.scale_to_unit(new_mean_shape[i], lm.get_tooth_centroid((new_mean_shape[i])))
            new_mean_shape[i] = lm.tooth_from_matrix_to_vector(lm.translate_to_origin(new_mean_shape[i]))
        
        # Terminating condition
        if (mean_shape - new_mean_shape < 1e-10).all():
            break
        
        mean_shape = new_mean_shape
    return mean_shape, aligned_shape
    
def hand_draw_box(number_samples,radiographs):
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

def perfect_fit_box(mean_shape,preprocessed_r,largest_b,box_param,toothnr):
    """  
    IMPORTANT NOTES REGARDING ESTIMATE
    - you have to spcify which tooth are you looking for
    - you have to spacify if it is upper or lower (as last parameter input upper/lower) -> this can be made dynamic
    """
    toothnr = 0
    estimates = []
    for rad_nr in range(number_samples): # number_samples
        e = estimate(rad_nr, mean_shape[toothnr], toothnr, preprocessed_r, largest_b, box_param, False)
	estimates.append(e)
	print('    Box for radiograph ' + str(rad_nr + 1) + ' done')    
    return estimates   
    
def fit_model(estimates,number_samples,number_teeth,mean_shape,radiographs,edges):
    detailed_boxes = []
    for i in range(number_samples):
        detailed_boxes.append(fit.get_detailed_boxes(estimates[i]))
    detailed_boxes = np.asarray(detailed_boxes)
    #print(detailed_boxes)

    # Get colours to print the teeth over the imgages 
    colors = visual.__get_colors(number_teeth)
    new_points = []
    # Loop over every image 
    for j in range(number_samples): # number_samples
        # Loop over every tooth 
        for i in range(number_teeth):
            # Apply the model over the boxes 
            img, newpoints = fit.fit_asm_model_to_box(mean_shape[i], detailed_boxes[j][i], radiographs[j], 1000, colors[i], edges[j])
            # Smooth the obtained points 
            newpoints = fit.smooth_model(newpoints)
            visual.render_model_over_image(newpoints, radiographs[j], i+1, colors[i], False)
            new_points.append(newpoints)
        visual.savefinalimage(radiographs[j], j + 1)
        print('* Fitting completed for image ' + str(j + 1) + ' *')
    
    
if __name__ == '__main__':
    
    # Global variables 
    n_components = 8
    draw_handboxes = False 
    # ***** Read the landmarks ******
    number_teeth = 8 
    number_samples = 14
    
    directory = 'original/landmarks'
    teeth_landmarks = load_landmarks(number_teeth,number_samples, directory)
    #directory = 'mirrored/landmarks'
    #teeth_landmarks.extend(load_landmarks(number_teeth,number_samples, directory))
    print('* Landmarks loaded *')
     
     
    # ***** Print the teeth_landmarks over radiographs ***** 
    #visual.print_landmarks_over_radiographs(teeth_landmarks)
    
    around_origin_scaled, mean_shape = pre_procrustes(number_teeth,number_samples,teeth_landmarks)
    
    print('* Starting Procrustes Analysis *')
    
    # ***** Do the Generalized Procrustes Analysis on the landmarks ***** 
    
    mean_shape, aligned_shape = procrustes(around_origin_scaled,number_teeth, mean_shape)
    
    
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
        pca_res.fit(aligned_shape[i])
        reduced_dim.append(pca_res.transform(aligned_shape[i])) # Reduce dimensionality of the training data

    reduced_dim = np.asarray(reduced_dim)
    
    
    # ***** Print the ASM of the landmarks *****
    """ Be careful when printing, the order is  k[i,1],k[i,0] not the other way around!"""
    #for i in range(number_teeth):
    #    visual.plot_procrustes(mean_shape[i],aligned_shape[i], i+1, True)       
       
    # ***** Do pre-processing of the images *****
    
    print('* Starting preprocessing *')
    # radiographs contains the raw radiographs images
    radiographs = pp.load_radiographs(number_samples,False)
    # Preprocess the images
    preprocessed_r = []
    for i in range(len(radiographs)):
        preprocessed_r.append(pp.preprocess_radiograph(radiographs[i]))
    # Find the edges of the radiographs
    edges = []
    for i in range(len(preprocessed_r)):
        # Finding the edges 
        edges.append(pp.togradient_sobel(preprocessed_r[i]))
    print('* Preprocessing finished *')

    # ***** Ask the user to draw the boxes around the jaws ***** 
    if draw_handboxes: 
        hand_draw_box(number_samples,radiographs)
    # mean_box = bx.get_mean_boxes(teeth_boxes) # dimension is 1x8

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
    print('* Finding upper refined boxes *')
    estimates = perfect_fit_box(mean_shape,preprocessed_r,largest_b,upper,0)
    print('* Finding lower refined boxes *')
    e2 = perfect_fit_box(mean_shape,preprocessed_r,largest_b,lower,5)
    
    for i in range(len(estimates)):
        estimates[i].extend(e2[i])
    print('* Found boxes for the teeth *') 
    
    # ***** Apply and fit the model over the image ***** 
    fit_model(estimates,2,number_teeth,mean_shape,radiographs,edges)
