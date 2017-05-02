# -*- coding: utf-8 -*-
import os
import numpy as np

class Landmark():
    def __init__(self):
        # Create the set of points
        self.points = []
        
    def readPoints(self, file_path):
        if file_path is None:
            print('File location is None.')
            return
            
        # Check that the file exists
        if not os.path.exists(file_path):
            print(file_path, ' does NOT not exist.')
            return
        
        print('******', file_path, '******')
        
        # Open the file and read all the lines 
        with open(file_path) as f:
            content = f.readlines()
            # Divide the content in a list of numbers. 
        content = [x.strip() for x in content]
        print(len(content))
        
        i = 0
        for i in range(0, len(content)-1): # -1 because then I add +1
            np.concatenate((self.points, np.array([float(content[i]), float(content[i+1])])), axis = 0) # DOESN'T WORK
        
        print(self.points)
                    
    def printLandmark(self):
        pass
        print('still TODO')