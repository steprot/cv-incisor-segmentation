# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import landmark as lm
from sklearn.decomposition import PCA
import preprocessing as pp
import boxes as bx
import fitter as fit
import visualise as visual
from estimate import estimate

def load_landmarks(number_teeth, number_samples, directory, mirrored):
    i = 1
    if mirrored:
        index = 14
    else:
        index = 0
    teeth_landmarks = [] # 3D array: 8*number_samples*80
    path ='../_Data/Landmarks/' + str(directory)
    while i <= number_teeth: # For all the different teeth 
        j = 1 
        landmark = []
        while j <= number_samples: # For the same tooth but for the different persons
            # Specify the name where to read the landmarks from
            directory = path + str(index + j) + '-' + str(i) + '.txt'
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
    
def pre_procrustes(number_teeth, number_samples, teeth_landmarks):
    print(number_samples)
    # ***** Scale landmarks ***** 
    scaled = []
    for i in range(number_teeth):
        scaled_row = []
        for j in range(number_samples):
            scaled_row.append(np.array(lm.tooth_from_matrix_to_vector(lm.scale_to_unit(lm.tooth_from_vector_to_matrix(teeth_landmarks[i, j]), \
                                                                      lm.get_tooth_centroid(teeth_landmarks[i,j])))))
        scaled.append(np.array(scaled_row))
    scaled = np.array(scaled)
    
    
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
    
    print('around_origin_scaled', around_origin_scaled.shape)
    print('mean_shape', mean_shape.shape)
    return around_origin_scaled, mean_shape
    
def procrustes(around_origin_scaled, number_teeth, mean_shape, number_samples):
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
    
def hand_draw_box(number_samples, radiographs):
    print('* Asking for hand drown boxes *')
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
    
def perfect_fit_box(isupper, test_preproc, preprocessed_r, number_samples, largest_b, box_param, nr, build_model):
    """  
    isupper - upper incisiors (True), lower incisors (False)
    preprocessed_r - preprocessed radiographs
    largest_b - parameter of the largest hand drawn box which we will fine tune
    box_param - array of hand drawn boxes either upper or lower 
    nr - 1 if we want to find box only on one radiograph, else the number of radiographs
    build_model - should the model be built (True) or loaded (False)
    """
    estimates = []
    e = estimate(test_preproc, isupper, preprocessed_r, 0, number_samples, largest_b, box_param, False, build_model)
    estimates.append(e)
    print('    Box for radiograph done')   
    return estimates  

#def perfect_fit_box(isupper, preprocessed_r, number_samples, largest_b, box_param, nr, build_model):
#    """  
#    isupper - upper incisiors (True), lower incisors (False)
#    preprocessed_r - preprocessed radiographs
#    largest_b - parameter of the largest hand drawn box which we will fine tune
#    box_param - array of hand drawn boxes either upper or lower 
#    nr - 1 if we want to find box only on one radiograph, else the number of radiographs
#    build_model - should the model be built (True) or loaded (False)
#    """
#    estimates = []
#    for rad_nr in range(nr): # number_samples
#        e = estimate(preprocessed_r[13], isupper, preprocessed_r, rad_nr, number_samples, largest_b, box_param, False, build_model)
#	estimates.append(e)
#	print('    Box for radiograph ' + str(rad_nr + 1) + ' done')    
#    return estimates   
    
def fit_model(estimates, nr, number_teeth, mean_shape, radiograph, edge, save=True):
    detailed_boxes = []
    for i in range(nr):
        detailed_boxes.append(fit.get_detailed_boxes(estimates[i]))
    detailed_boxes = np.asarray(detailed_boxes)
    #print(detailed_boxes)

    # Get colours to print the teeth over the imgages 
    colors = visual.get_colors(number_teeth)
    new_points = []
    # Loop over every image 
    for j in range(nr): # number_samples
        new_landmarks_sample = []
        # Loop over every tooth 
        for i in range(number_teeth):
            # Apply the model over the boxes 
            #print(detailed_boxes[j][i])
            img, newpoints = fit.fit_asm_model_to_box(mean_shape[i], detailed_boxes[j][i], radiograph, 1000, colors[i], edge, i)
            # Smooth the obtained points 
            newpoints = fit.smooth_model(newpoints)
            visual.render_model_over_image(newpoints, radiograph, i+1, colors[i], False)
            
            tmp = []
            for i in (range(newpoints.shape[0])):
                tmp.append(newpoints[i, 0])
                tmp.append(newpoints[i, 1])
            newpoints = np.asarray(tmp)
            new_landmarks_sample.append(np.asarray(newpoints))
            
        if save:
            visual.save_final_image(radiograph, TESTINGON + 1)
        print('    Fitting completed for image ' + str(TESTINGON + 1)) 
        
        new_landmarks_sample = np.asarray(new_landmarks_sample)      
        new_points.append(new_landmarks_sample.T)
        
    new_points = np.asarray(new_points)
    return new_points
    
def estimate_sse(number_teeth, nr, new_landmarks, test_landmark):
    for j in range(nr): # number_samples 
        ssex_sample = 0
        ssey_sample = 0
        ssexabs_sample = 0
        sseyabs_sample = 0
        for i in range(number_teeth):
            x, y, xabs, yabs = lm.sse_tooth(new_landmarks[j, :, i], test_landmark[i, :])
            ssex_sample += x
            ssey_sample += y
            ssexabs_sample = xabs
            sseyabs_sample = yabs
    return ssex_sample/40, ssey_sample/40, ssexabs_sample/40, sseyabs_sample/40

   
      
         

#   ************************************* MAIN **********************************************************************
if __name__ == '__main__':
    
    # Global variables 
    n_components = 8
    draw_handboxes = False 
    TESTINGON = 8
    
    
    # ***** Read the landmarks ******
    number_teeth = 8 
    number_samples = 14
    
    directory = 'original/landmarks'
    teeth_landmarks = load_landmarks(number_teeth, number_samples, directory, False)
    #directory = 'mirrored/landmarks'
    #teeth_mirrored = load_landmarks(number_teeth, number_samples, directory, True)
    #teeth_landmarks = np.concatenate((teeth_landmarks, teeth_mirrored), axis=1)
    ## Double the number of samples because of the mirrored ones 
    #number_samples *= 2
    #print('* Landamrks loaded *')
     
    # ***** Print the teeth_landmarks over radiographs ***** 
    #visual.print_landmarks_over_radiographs(teeth_landmarks)
    
    # Removing the nth element to test on it 
    test_landmark = teeth_landmarks[:,TESTINGON,:]
    #print('test_landmark', test_landmark.shape)
    teeth_landmarks = np.delete(teeth_landmarks, TESTINGON, 1)
    #print('teethlandmarks', teeth_landmarks.shape)
    
    around_origin_scaled, mean_shape = pre_procrustes(number_teeth, number_samples-1, teeth_landmarks) # number_samples - 1 because of testing 
    print('* Starting Procrustes Analysis *')
    
    # ***** Do the Generalized Procrustes Analysis on the landmarks ***** 
    mean_shape, aligned_shape = procrustes(around_origin_scaled, number_teeth, mean_shape, number_samples-1) 
    
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
    #print(number_samples)
    radiographs = pp.load_radiographs(number_samples, False)
    #print(len(radiographs[0].shape))
    
    #mirrored = lm.get_mirrored_radiographs(radiographs, False)
    #for i in range(len(mirrored)):
    #    radiographs.append(mirrored[i])
    
    ''' It is possible to do the saving of the preprocessed images with
        the pickle library in order to save time  '''
    
    test_radio = radiographs[TESTINGON]
    del radiographs[TESTINGON]
    test_preprocessed = pp.preprocess_radiograph(test_radio)
    test_edge = pp.togradient_sobel(test_preprocessed)
    
    # Preprocess the images
    preprocessed_r = []
    #print(len(radiographs))
    for i in range(len(radiographs)):
        preprocessed_r.append(pp.preprocess_radiograph(radiographs[i]))
    # Find the edges of the radiographs
    print('    Finished preprocessing')
    print('    Finding edges')
    edges = []
    for i in range(len(preprocessed_r)):
        # Finding the edges 
        edges.append(pp.togradient_sobel(preprocessed_r[i]))
    print('* Preprocessing finished *')
    
    # ***** Ask the user to draw the boxes around the jaws ***** 
    if draw_handboxes: 
        hand_draw_box(number_samples, radiographs)
    # mean_box = bx.get_mean_boxes(teeth_boxes) # dimension is 1x8


    # ***** Reading the boxes from file and get the largest one *****
    # Getting the largest boxes from the file     
    boxes_from_file = bx.read_boxes_from_file()   
    
    #remove for testing 
    boxes_from_file = np.delete(boxes_from_file, TESTINGON, 0)

    largest_b = bx.get_largest_boxes(boxes_from_file)
    upper = []
    lower = []
    for i in range(len(boxes_from_file)):
        upper.append([boxes_from_file[i][0], boxes_from_file[i][1], boxes_from_file[i][2], boxes_from_file[i][3]])
        lower.append([boxes_from_file[i][4], boxes_from_file[i][5], boxes_from_file[i][6], boxes_from_file[i][7]])
    print('* Largest box obtained *')
    
    #for i in range(number_samples):
    #    bx.print_boxes_on_teeth(largest_b, radiographs[i])
    
    ''' Number of inputs to estimate and fit '''
    NR_INPUT = 1
    
    # ***** Sharp the boxes ***** 
    print('* Finding upper refined boxes *')
    estimates = perfect_fit_box(True, test_preprocessed, preprocessed_r, number_samples-1, largest_b, upper, NR_INPUT, True)
    print('* Finding lower refined boxes *')
    e2 = perfect_fit_box(False, test_preprocessed, preprocessed_r, number_samples-1, largest_b, lower, NR_INPUT, True)
    
    for i in range(len(estimates)):
        estimates[i].extend(e2[i])
    print('* Found boxes for the teeth *') 
    
    # ***** Apply and fit the model over the image ***** 
    new_landmarks = fit_model(estimates, NR_INPUT, number_teeth, mean_shape, test_radio, test_edge, True)
    cv2.destroyAllWindows()
        
    # ***** Estimate error with the Mean Squared Errors *****
    print('* Estimating MSE *')
    msex, msey , msexabs, mseyabs= estimate_sse(number_teeth, NR_INPUT, new_landmarks, test_landmark)
    msex = msex/8
    msey = msey/8
    msexabs = msexabs/8
    mseyabs = mseyabs/8
    print('  Mean Squared Errors per estimations: ')
    print('  {:.2f}, {:.2f}'.format(msex, msey))
    print('  Mean Absolute Error with respect to image widht and heights: ')
    print('  {:.2f} ( {:.2f}% ), {:.2f} ( {:.2f}% )'.format(msexabs, 100*msexabs/test_radio.shape[1], mseyabs, 100*mseyabs/test_radio.shape[0]))

    
    