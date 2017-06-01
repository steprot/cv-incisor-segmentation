import colorsys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import landmark as lm
import os

SCREEN_H = 800
SCREEN_W = 1200

number_teeth = 8 
number_samples = 14

def print_landmarks_over_radiographs(teeth_landmarks):
    '''
    Prints teeth's landmark points over the radiographs.
    Parameters: 
         teeth_landmarks; 3D table containing the 80 landmark points per tooth, per sample.
    '''
    i = 0
    while i < number_samples:
        j = 0
        directory = '../_Data/Radiographs/' + str("%02d" % (i+1)) + '.tif'
        dir_path = os.path.join(os.getcwd(), directory)
        img = cv2.imread(dir_path)
        while j < number_teeth:
            render_landmark_over_image(img, teeth_landmarks[j,i,:])
            # Go to the next tooth
            j += 1
        # Got the next sample   
        i += 1

def render_landmark_over_image(img, landmark):
    
    points = lm.tooth_from_vector_to_matrix(landmark)
    
    for i in range(len(points) - 1):
        cv2.line(img, (int(points[i, 1]), int(points[i, 0])), (int(points[i + 1, 1]), int(points[i + 1, 0])), (0, 255, 0))

    # img = __fit_on_screen(img)
    cv2.imshow('Rendered image', img) # SCREEN FITTING 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def render_model_over_image(points, img, incisor_nr, color, save): 
    '''
    Render the model (stored differently from the landmarks, over an image)
    Parameters:
        points; the model to display
        img: image over which display the model
        incisor_nr; number of the tooth corresponding to the model 
        color; the color with which print 
        save; boolean, if true it saves the image in the folder  
    '''   
    for i in range(len(points) - 1):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])), (int(points[i + 1, 0]), int(points[i + 1, 1])), color)
    if save: 
        # Specify the name where to print the Procrustes images 
        directory = '../Plot/Final/finalresult' + str("%02d" % (incisor_nr))+'.png'
        dir_path = os.path.join(os.getcwd(), directory)
        cv2.imwrite(dir_path, img)
    cv2.imshow('finalresult' + str("%02d" % (incisor_nr))+'.png', img) # SCREEN FITTING 
    cv2.waitKey(0)
    
def save_final_image(img, i):
    directory = '../Plot/Final/finalresult' + str("%02d" % (i))+'.png'
    dir_path = os.path.join(os.getcwd(), directory)
    cv2.imwrite(dir_path, img)
    return

#def render_landmark_over_image(img, landmark):
#    
#    #points = lm.tooth_from_vector_to_matrix(landmark)
#    img = np.ones((1000, 600, 3), np.uint8) * 255
#    mean_shape = scale_for_print(landmark,1100)
#    points = translate(mean_shape,np.array([300,500]))
#    
#    for i in range(len(points) - 1):
#        #cv2.line(img, (int(points[i, 1]), int(points[i, 0])), (int(points[i + 1, 1]), int(points[i + 1, 0])), (0, 255, 0))
#        cv2.line(img, (int(points[i, 1]), int(points[i, 0])),
#                 (int(points[(i + 1) % 40, 1]), int(points[(i + 1) % 40, 0])),
#                 (0, 0, 0), 2)
#    img = __fit_on_screen(img)
#    cv2.imshow('Rendered image', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
def render_landmarks(data_collector):
    '''
    Method visualizes landmark points in a given data_collector of type DataCollector
    '''

    points = data_collector.as_matrix()
    max_y = points[:, 0].max()
    min_y = points[:, 0].min()
    max_x = points[:, 1].max()
    min_x = points[:, 1].min()

    img = np.zeros((int((max_y - min_y) * 1.1), int((max_x - min_x) * 1.1)))

    for i in range(len(points)):
        img[int(points[i, 0] - min_y), int(points[i, 1] - min_x)] = 1

    cv2.imshow('Rendered shape', img)
    cv2.waitKey(0)

def scale_for_print(array, value):
    centroid = np.mean(array, axis=0)
    points = (array - centroid).dot(value) + centroid
    return points        
    
def translate(trans, centroid):
    """
    Translates the landmark points so that the centre of gravitiy of this
    shape is at 'centroid'.
    """
    i = 0
    points = []
    while i<len(trans)-1:
        points.append([trans[i], trans[i+1]])
        i = i+2
    points = points + centroid
    
    return points
                        
def plot_procrustes(mean_shape, aligned_shapes, incisor_nr, save):
    '''
    Plots the result of the procrustes analysis.
    Parameters:
        mean_shape; mean shape for each tooth
        aligned_shapes; 14 for each tooth
        incisor_nr; index of tooth
        save(bool); whether to save the plot.
    '''
    
    # white background
    img = np.ones((1000, 600, 3), np.uint8) * 255
    mean_shape = scale_for_print(mean_shape,1100)
    points = translate(mean_shape, np.array([400,300]))

    for i in range(len(points)):
        cv2.line(img, (int(points[i, 1]), int(points[i, 0])),
                 (int(points[(i + 1) % 40, 1]), int(points[(i + 1) % 40, 0])),
                 (0, 0, 0), 2)
    # center of mean shape
    cv2.circle(img, (400,300), 10, (255, 255, 255))

    # plot aligned shapes in different colors
    colors = get_colors(len(aligned_shapes))
    
    for ind, aligned_shape in enumerate(aligned_shapes):
        aligned_shape =  scale_for_print(aligned_shape, 1100)
        points = translate(aligned_shape, np.array([400,300]))
        #points = lm.tooth_from_vector_to_matrix(aligned_shape)
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 1]), int(points[i, 0])),
                     (int(points[(i + 1) % 40, 1]), int(points[(i + 1) % 40, 0])),
                     colors[ind])

    # Show
    img = __fit_on_screen(img)
    cv2.imshow('Procrustes result for incisor ' + str("%02d" % (incisor_nr + 1)), img)
    cv2.waitKey(0)
    if save:
        # Specify the name where to print the Procrustes images 
        directory = '../Plot/Procrustes/' + str("%02d" % (incisor_nr + 1)) +'.png'
        dir_path = os.path.join(os.getcwd(), directory)
        cv2.imwrite(dir_path, img)
    cv2.destroyAllWindows()

#def print_landmarks(tooth_landmarks):
#    '''
#    Prints the single tooth landmark points over a black image. 
#    Parameter: 
#        tooth_landmarks; array containing the 80 landmark points of the tooth. 
#    '''
#    
#    # Transform the array in a matrix 
#    i = 0
#    points = []
#    while i < len(tooth_landmarks)-1: # -1 because then I add +1
#        points.append([float(tooth_landmarks[i]), float(tooth_landmarks[i+1])])
#        i += 2 # To go to the next couple
#    points = np.array(points)
#
#    max_y = points[:, 0].max()
#    min_y = points[:, 0].min()
#    max_x = points[:, 1].max()
#    min_x = points[:, 1].min()
#    
#    img = np.zeros((int((max_y - min_y) * 1.1), int((max_x - min_x) * 1.1)))
#    
#    for i in range(len(points)):
#        img[int(points[i, 0] - min_y), int(points[i, 1] - min_x)] = 1
#    img = __fit_on_screen(img)
#    cv2.imshow('Rendered shape', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
#
#def render_landmark(landmark):
#        
#    points = lm.tooth_from_vector_to_matrix(landmark)
#    
#    max_y = points[:, 0].max()
#    min_y = points[:, 0].min()
#    max_x = points[:, 1].max()
#    min_x = points[:, 1].min()
#    
#    img = np.zeros((int((max_y - min_y) * 1.1), int((max_x - min_x) * 1.1)))
#    
#    for i in range(len(points)):
#        img[int(points[i, 0] - min_y), int(points[i, 1] - min_x)] = 1
#    
#    cv2.imshow('Rendered shape', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def get_colors(num_colors):
    '''
    Get a list with ``num_colors`` different colors.
    Parameters:
        num_colors (int): The number of colors needed.
    Returns:
        list: ``num_colors`` different rgb-colors ([0,255], [0,255], [0,255])

    .. _Code based on:
        http://stackoverflow.com/a/9701141
    '''
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in colors]
    
def __fit_on_screen(image):
    '''
    Rescales the given image such that it fits on the screen.
    Parameters:
        image: The image to rescale
    '''
    # find minimum scale to fit image on screen
    scale = min(float(SCREEN_W) / image.shape[1], float(SCREEN_H) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    
