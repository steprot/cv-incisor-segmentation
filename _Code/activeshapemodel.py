# -*- coding: utf-8 -*-
import cv2
import cv2.cv as cv
import os
import numpy as np
import fnmatch

class ActiveShapeModel(object):
    
    def __init__(self, teeth_landmarks, number_samples):
        
        self.num_samples = number_samples
        self.points = teeth_landmarks
            
    def compute_mean(self):
        self.points.resize((self.num_samples, 80))       
        mean_points = np.mean(self.points, axis = 0)
        
        return mean_points
        