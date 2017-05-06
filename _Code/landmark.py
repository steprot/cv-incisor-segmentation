# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch

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
        
        return np.hstack(self.points)
        
    def as_matrix(self):
        
        return self.points
        
def tooth_from_vector_to_matrix(vector):
    # Transform the array in a matrix 
    i = 0
    points = []
    while i < len(vector)-1: # -1 because then I add +1
        points.append([float(vector[i]), float(vector[i+1])])
        i += 2 # To go to the next couple
    points = np.array(points)
    return points
    
def tooth_from_matrix_to_vector(vector):
    
    # Transform the matrix in an array 
    points = np.hstack(vector)
    return points
                                                    
def compute_centroids(tooth):
    """ 
    Compute the centroid as the mean point of all the teeth, and then the mean
    value of x and y of the mean point 
    """
    mean_tooth = np.mean(tooth, axis = 0)
    mean_tooth = tooth_from_vector_to_matrix(mean_tooth)
    centroid = np.mean(mean_tooth, axis = 0,)
    return centroid 

def translate_to_origin(landmarks):
    """
    Translates the landmark points so that the centre of gravitiy of this
    shape is at the origin.
    """
    i = 0
    points = []
    while i<len(landmarks)-1:
        points.append([landmarks[i],landmarks[i+1]])
        i = i+2
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    return points
    
def scale_to_unit(tooth, centroid):
    
    scale_factor = np.sqrt(np.power(tooth - centroid, 2).sum())
    points = tooth.dot(1. / scale_factor)
    
    return points
    
def align_teeth_to_mean_shape(elem, mean):
    align_parameters(elem, mean)
    pass
    
def align_parameters(element, scaled_mean):
    """
    Finds the best parameters to align the two shapes. 
    Based on: "An Introduction to Active Shape Model"
    Parameters:
        element; n_teeth*80
        scaled_mean; 
    """
    l1 = len(element)/2
    l2 = len(scaled_mean)/2

    # make sure both shapes are mean centered for computing scale and rotation
    x1_centroid = np.array([np.mean(element[:l1]), np.mean(element[l1:])])
    x2_centroid = np.array([np.mean(scaled_mean[:l2]), np.mean(scaled_mean[l2:])])
    element = [x - x1_centroid[0] for x in element[:l1]] + [y - x1_centroid[1] for y in element[l1:]]
    scaled_mean = [x - x2_centroid[0] for x in scaled_mean[:l2]] + [y - x2_centroid[1] for y in scaled_mean[l2:]]
    
    # a = (x1.x2)/|x1|^2
    norm_x1_sq = (np.linalg.norm(element)**2)
    a = np.dot(element, scaled_mean) / norm_x1_sq

    # b = sum_1->l2(x1_i*y2_i - y1_i*x2_i)/|x1|^2
    b = (np.dot(element[:l1], scaled_mean[l2:]) - np.dot(element[l1:], scaled_mean[:l2])) / norm_x1_sq

    # s^2 = a^2 + b^2
    s = np.sqrt(a**2 + b**2)

    # theta = arctan(b/a)
    theta = np.arctan(b/a)

    # the optimal translation is chosen to match their centroids
    t = x2_centroid - x1_centroid
    
    print('result of the align parametersss', t, s, theta)
    return t, s, theta

def render_landmark(landmark):
        
    points = tooth_from_vector_to_matrix(landmark)
    
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
        
def render_landmark_over_image(img, landmark):
    
    points = tooth_from_vector_to_matrix(landmark)
    
    for i in range(len(points) - 1):
        cv2.line(img, (int(points[i, 1]), int(points[i, 0])), (int(points[i + 1, 1]), int(points[i + 1, 0])), (0, 255, 0))

    height = 500
    scale = height / float(img.shape[0])
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('rendered image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rendered image', window_width, window_height)
    cv2.imshow('rendered image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img
    