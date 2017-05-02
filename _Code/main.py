# -*- coding: utf-8 -*-
import os
import numpy as np
import fnmatch
from landmark import Landmark

if __name__ == '__main__':
    
    # Read the landmarks       
    # Specify the name where to read the landmarks from
    dir_path = os.path.join(os.getcwd(), '../_Data/Landmarks/original/landmarks1-1.txt')
    
    # Create a landmarks object
    # Each landmark is a tooth from a single person 
    landmarks = Landmark()
    
    # Read the points of the tooth
    landmarks.readPoints(dir_path)
    
    landmarks.printLandmark()