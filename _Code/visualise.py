import colorsys
import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec


SCREEN_H = 800
SCREEN_W = 1200

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

    cv2.namedWindow('Rendered image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Rendered image', window_width, window_height)
    cv2.imshow('Rendered image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img

def print_image(img, title="Radiograph"):
    '''
        Shows the given image.
    '''
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plot_procrustes(mean_shape, aligned_shapes, incisor_nr=0, save=False):
    '''
        Plots the result of the procrustes analysis.
    
        Args:
            mean_shape (Landmarks): The mean shape.
            aligned_shapes ([Landmarks]): The other shapes, aligned with the mean.
            incisor_nr (int): Index of the corresponding incisor.
            save(bool): Whether to save the plot.
    '''
    
    # white background
    img = np.ones((1000, 600, 3), np.uint8) * 255

    # plot mean shape
    mean_shape = mean_shape.scale(1500).translate([300, 500])
    points = mean_shape.as_matrix()
    for i in range(len(points)):
        cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                 (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                 (0, 0, 0), 2)
    ## center of mean shape
    cv2.circle(img, (300, 500), 10, (255, 255, 255))

    # plot aligned shapes
    colors = __get_colors(len(aligned_shapes))
    for ind, aligned_shape in enumerate(aligned_shapes):
        aligned_shape = aligned_shape.scale(1500).translate([300, 500])
        points = aligned_shape.as_matrix()
        for i in range(len(points)):
            cv2.line(img, (int(points[i, 0]), int(points[i, 1])),
                     (int(points[(i + 1) % 40, 0]), int(points[(i + 1) % 40, 1])),
                     colors[ind])

    # show
    img = __fit_on_screen(img)
    cv2.imshow('Procrustes result for incisor ' + str(incisor_nr), img)
    cv2.waitKey(0)
    if save:
        cv2.imwrite('Plot/Procrustes/'+str(incisor_nr)+'.png', img)
    cv2.destroyAllWindows()


def __get_colors(num_colors):
    """Get a list with ``num_colors`` different colors.

    Args:
        num_colors (int): The number of colors needed.

    Returns:
        list: ``num_colors`` different rgb-colors ([0,255], [0,255], [0,255])

    .. _Code based on:
        http://stackoverflow.com/a/9701141
    """
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return [(int(r*255), int(g*255), int(b*255)) for (r, g, b) in colors]
    
def __fit_on_screen(image):
    """Rescales the given image such that it fits on the screen.

    Args:
        image: The image to rescale.

    Returns:
        The rescaled image.

    """
    # find minimum scale to fit image on screen
    scale = min(float(SCREEN_W) / image.shape[1], float(SCREEN_H) / image.shape[0])
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    
