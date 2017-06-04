# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import math

class Landmarks():
    
    def __init__(self, source):
        
        # Create the set of points
        self.points = []   
        if source is not None:
            self.read_landmark(source)
        
    def read_landmark(self, input_file):
        
        # Checkt that the input file is not None
        if input_file is None:
            print('Input file location is None.')
            return
            
        # Check that the input file exists
        if not os.path.exists(input_file):
            print('Input file does not exist.')
            return
       
        ## Open the file and read all the lines 
        #with open(input_file) as f:
        #    content = f.readlines()
        #
        ## Divide the content in a list of numbers. 
        #content = [x.strip() for x in content]
        #                
        #i = 0
        #points = []
        #while i < len(content)-1: # -1 because then I add +1
        #    points.append([float(content[i]), float(content[i+1])])
        #    i += 2 # To go to the next couple 
            
        tmp = open(input_file).readlines()
        for ind in range(0, len(tmp), 2):
            self.points.append(np.array([float(tmp[ind + 1].strip()), float(tmp[ind].strip())]))
                         
        self.points = np.array(self.points)

    def as_vector(self):
        #Possibly can be removed
        return np.hstack(self.points)
        
    def as_matrix(self):
        #Possibly can be removed -- Pretty sure this is wrong
        return self.points
        
def tooth_from_vector_to_matrix(vector):
    # Transform the array into a matrix 
    """Original: [ x1, y1, x2, y2, x3, y3 ... ] 
       Result  : [ [x1 y1] ,  [x2 y2],  [x3 y3] ... ]
    """
    i = 0
    points = []
    while i < len(vector)-1: # -1 because then I add +1
        points.append([float(vector[i]), float(vector[i+1])])
        i += 2 # To go to the next couple
    points = np.array(points)
    return points
    
def tooth_from_matrix_to_vector(vector):
    """Original: [ [x1 y1] ,  [x2 y2],  [x3 y3] ... ] 
       Result  : [ x1, y1, x2, y2, x3, y3 ... ] """
    # Transform the matrix in an array by brushing the first list together    
    i = 0
    points = []

    x=len(vector)

    while i < x:
        points.append(float(vector[i, 0]))
        points.append(float(vector[i, 1]))
        i += 1 
    
    return points

def get_tooth_centroid(points):
    """
        Returns the center of mass [x, y] of this shape.  
    """
    return np.mean(points, axis=0)

def get_tooth_center(points):
    """
        Returns the center [x,y] of this shape.
    """
    vertical_size = points[:, 1].min() + (points[:, 1].max() - points[:, 1].min())/2
    horizontal_size = points[:, 0].min() + (points[:, 0].max() - points[:, 0].min())/2
    return [horizontal_size, vertical_size]

def translate_to_origin(landmarks):
    '''
        Translates the landmark points so that the centre of gravitiy of this
        shape is at the origin.
    '''
    i = 0
    points = []
    while i<len(landmarks)-1:
        points.append([landmarks[i], landmarks[i+1]])
        i = i+2
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    return points
    
def scale_to_unit(tooth, centroid):
    '''
        Scale the landmark, so that the norm of the shape is 1
    '''
    scale_factor = np.sqrt(np.power(tooth - centroid, 2).sum())
    points = tooth.dot(1. / scale_factor)
    
    return points
    
def rotate(element, theta):
    '''
        Rotate element by theta
    '''
    # create rotation matrix
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    
    element = tooth_from_vector_to_matrix(element)
    rotated_element = np.zeros_like(element)
    for ind in range(len(element)):
        rotated_element[ind, :] = element[ind, :].dot(rotation_matrix)
    
    return rotated_element
    
def scale(element, value):
    '''
        Rescale element by given value
    '''
    e = element.dot(value)

    return e
 
def align_teeth_to_mean_shape(elem, mean):
    ''' 
    Allign current incisor to the mean shape
    '''
    t, s, theta = get_aligning_parameters(elem, mean)
    rotated_element = rotate(elem, theta)
    scaled = scale(rotated_element, s)
    
    # project into tangent space by scaling x1 with 1/(x1.x2)
    # tangent space of a manifold facilitates the generalization of vectors from 
    # affine spaces to general manifolds, since in the latter case one cannot 
    # simply subtract two points to obtain a vector that gives the displacement 
    # of the one point from the other

    xx = np.dot(tooth_from_matrix_to_vector(scaled), mean)
    result = (tooth_from_matrix_to_vector(scaled))/xx
    
    return result
    
    
def get_aligning_parameters(element, scaled_mean):
    """
        Finds the best parameters to align the two shapes. 
        Based on: "An Introduction to Active Shape Model"
        Parameters:
            element; 1*80
            scaled_mean; 1*80
            
        When we want to scale and rotate element by (s, theta) the optimal way ( minimizing 
        |s*A*element - scaled_mean|). 
        
        --- A performs a rotation and of element by theta)
        --- s^2 = a^2+b^2
            a = element*scaled_mean/|element|^2
            b = sum( [element(i,0)*scaled_mean(i,1) - element(i,1)*scaled_mean(i,0)]/|element|^2)
            sigma = arctan(b/a)
    """
    l1 = len(element)/2
    l2 = len(scaled_mean)/2

    # make sure both shapes are mean centered for computing scale and rotation
    x1_centroid = np.array([np.mean(element[:l1]), np.mean(element[l1:])])
    x2_centroid = np.array([np.mean(scaled_mean[:l2]), np.mean(scaled_mean[l2:])])
    element = [x - x1_centroid[0] for x in element[:l1]] + [y - x1_centroid[1] for y in element[l1:]]
    scaled_mean = [x - x2_centroid[0] for x in scaled_mean[:l2]] + [y - x2_centroid[1] for y in scaled_mean[l2:]]
    
    # a = element*scaled_mean/|element|^2
    norm_x1_sq = (np.linalg.norm(element)**2)
    a = np.dot(element, scaled_mean) / norm_x1_sq

    # b = sum( [element(i,0)*scaled_mean(i,1) - element(i,1)*scaled_mean(i,0)]/|element|^2)
    b = (np.dot(element[:l1], scaled_mean[l2:]) - np.dot(element[l1:], scaled_mean[:l2])) / norm_x1_sq

    # s^2 = a^2 + b^2
    s = np.sqrt(a**2 + b**2)

    # theta = arctan(b/a)
    theta = np.arctan(b/a)

    # the optimal translation is chosen to match their centroids
    t = x2_centroid - x1_centroid
    
    #print('result of the align parameters', t, s, theta)
    return t, s, theta
    
#def compute_new_mean_shape(aligned_teeth, number_teeth, number_samples):
#    '''
#        Compute the new mean, starting from the just aligned teeth.
#        aligned_teeth is a list of 24 arrays, each with 80 elements (the landmark points)
#    '''
#    # print('aligned teeth shape', aligned_teeth.shape)
#    
#    new_mean = []
#    for i in range(0, number_teeth):
#        new_mean.append(np.array(np.mean(aligned_teeth[i], axis = 0)))
#    new_mean = np.array(new_mean)
#    
#    # print('new_mean shape', new_mean.shape)
#    return new_mean

#def lists_to_matrix(data, number_teeth, number_samples):
#    ''' '''
#    res = []
#    for i in range(number_teeth):
#        row = []
#        for j in range(number_samples):
#            row.append(tooth_from_vector_to_matrix(data[i, j]))
#        res.append(np.array(row))
#    res =  np.array(res)
#    return res

#def matrix_to_list(data, number_teeth, number_samples):
#    res = []
#    for i in range(number_teeth):
#        row = []
#        for j in range(number_samples):
#            row.append(tooth_from_matrix_to_vector(data[i, j]))
#        res.append(np.array(row))
#    res =  np.array(res)
#    return res
    
def get_mirrored_radiographs(radiographs, save):
    '''
    Procude the mirrored images from the original ones.
    Parameters:
        radiographs; list of images
        save (bool); whether to save the mirrored images or not. 
    '''
    mirrored = []
    for i in range(len(radiographs)):
        (ih, iw, c) = radiographs[i].shape
        image = radiographs[i].copy()
        for j in range(ih):
            image[j,:] = image[j,::-1]
        mirrored.append(image)
        if save:
            # Specify the name where to print the Procrustes images 
            directory = '../Plot/Mirrored/mirrored' + str("%02d" % (i+1)) +'.tif'
            dir_path = os.path.join(os.getcwd(), directory)
            cv2.imwrite(dir_path, image)  
    return mirrored
    
def sse_tooth(tooth1, tooth2):
    i = 0
    #sse = 0
    ssex = 0
    ssey = 0
    while i < len(tooth1)-1:
        #sse += math.sqrt((tooth1[i] - tooth2[i+1])**2 + (tooth1[i+2] - tooth2[i+3])**2)
        ssex += math.sqrt((tooth1[i] - tooth2[i])**2)
        ssey += math.sqrt((tooth1[i+1] - tooth2[i+1])**2)
        i += 2
    return ssex, ssey