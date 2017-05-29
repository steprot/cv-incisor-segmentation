import math
import sys
import cv2
import numpy as np
from landmark import get_tooth_center, tooth_from_vector_to_matrix, get_tooth_centroid
    
def get_detailed_boxes(boxes):
    offset = 0
    # Don't have to print them, but create a structure with all the detailed boxes 
    # Do it for the upper structure 
    lx = boxes[0]
    rx = boxes[2]
    width = rx - lx
    width = int(width / 4)
    width2 = 2 * width
    box = []
    box.append([lx - offset, boxes[1], lx + width + offset, boxes[3]])
    box.append([lx + width - offset, boxes[1], lx + width2 + offset, boxes[3]])
    box.append([lx + width2 - offset, boxes[1], rx - width + offset, boxes[3]])
    box.append([rx - width - offset, boxes[1], rx + offset, boxes[3]])
    # Do it for the lower structure 
    lx = boxes[4]
    rx = boxes[6]
    width = rx - lx
    width = int(width / 4)
    width2 = 2 * width
    box.append([lx - offset, boxes[5], lx + width + offset, boxes[7]])
    box.append([lx + width - offset, boxes[5], lx + width2 + offset, boxes[7]])
    box.append([lx + width2 - offset, boxes[5], rx - width + offset, boxes[7]])
    box.append([rx - width - offset, boxes[5], rx + offset, boxes[7]])
    
    return np.asarray(box)
    
def fit_asm_model_to_box(toothmodel, toothbox, radiograph, fact, color):
    # Moving the model and centering it in the middle of the box 
    print(toothbox)
    toothmodel = toothmodel * fact
    points = tooth_from_vector_to_matrix(toothmodel)
    # Swap X and Y
    for i in range(len(points)):
        tmp = points[i, 0]
        points[i, 0] = points[i, 1]
        points[i, 1] = tmp
        
    box_center = np.asarray([(toothbox[2] + toothbox[0])/2, (toothbox[3] + toothbox[1])/2])
    #print('box_center' ,box_center)
    tooth_center = np.asarray(get_tooth_center(points))
    #print('tooth_center', tooth_center)
    translation_vec = box_center - tooth_center
    #print(translation_vec)
    points = points + translation_vec
    
    # Scale the model, to make it perfectly fit the box 
    hmax = points[:,1].max()
    hmin = points[:,1].min()
    wmax = points[:,0].max()
    wmin = points[:,0].min()
    bheight =toothbox[3] - toothbox[1]
    bwidth = toothbox[2] - toothbox[0]
    hfactor = bheight/(hmax - hmin)
    wfactor = bwidth/(wmax - wmin)
    #print('hfactor and wfactor', hfactor, wfactor)
    centroid = get_tooth_centroid(points)
    #print('centroid', centroid)
    ypoints = points[:,1]
    ypoints = (ypoints - centroid[1]).dot(hfactor) + centroid[1]  
    xpoints = points[:, 0]
    xpoints = (xpoints - centroid[0]).dot(wfactor) + centroid[0] 
    #print('ypoints', ypoints)
    #print('xpoints', xpoints)
    points = np.stack((xpoints, ypoints), axis=-1)
        
    tooth_center = np.asarray(get_tooth_center(points))
    #print('tooth_center', tooth_center)
    translation_vec = box_center - tooth_center
    #print(translation_vec)
    points = points + translation_vec
    
    # Printing the model over the radiograph
    img = radiograph.copy()  
    cv2.circle(img, (int(box_center[0]), int(box_center[1])), 1, (255, 255, 255), 2)
    #cv2.circle(img, (int(box_center[0]), int(hmax)), 1, (0, 0, 255), 2)
    #cv2.circle(img, (int(box_center[0]), int(hmin)), 1, (0, 0, 255), 2)
    #cv2.circle(img, (int(wmax), int(box_center[1])), 1, (0, 0, 255), 2)
    #cv2.circle(img, (int(wmin), int(box_center[1])), 1, (0, 0, 255), 2)
    
    for i in range(len(points)):
        cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), 1, color, 2)
        #cv2.line(img, (int(points[i, 1]), int(points[i, 0])),
        #         (int(points[(i + 1) % 40, 1]), int(points[(i + 1) % 40, 0])),
        #         (0, 255, 0), 2)

    # Show the image 
    cv2.imshow('Model over radiograph ', img)
    cv2.waitKey(0)    
    
    return img 
