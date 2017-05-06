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
                                                    
def compute_mean(tooth):
    """
    Computes the mean for a tooth, over all the samples stored 
    Parameter:
        tooth; number_sample * 80 matrix, containing 80 landmark points for each sample 
    """
    t = tooth.transpose()
    mean_points = np.mean(t, axis = 0)
    return mean_points

def render_landmark(landmark):
        
    # Transform the array in a matrix 
    i = 0
    points = []
    while i < len(landmark)-1: # -1 because then I add +1
        points.append([float(landmark[i]), float(landmark[i+1])])
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
        
def render_landmark_over_image(img, landmark):
    
    # Transform the array in a matrix 
    i = 0
    points = []
    while i < len(landmark)-1: # -1 because then I add +1
        points.append([float(landmark[i]), float(landmark[i+1])])
        i += 2 # To go to the next couple
    points = np.array(points)
    
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