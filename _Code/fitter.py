import math
import sys
import cv2
import numpy as np
import landmark as lm
    
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
    
def fit_asm_model_to_box(toothmodel, toothbox, radiograph, fact, color, edge_img):
    # Moving the model and centering it in the middle of the box 
    #print(toothbox)
    toothmodel = toothmodel * fact
    points = lm.tooth_from_vector_to_matrix(toothmodel)
    # Swap X and Y
    for i in range(len(points)):
        tmp = points[i, 0]
        points[i, 0] = points[i, 1]
        points[i, 1] = tmp
        
    box_center = np.asarray([(toothbox[2] + toothbox[0])/2, (toothbox[3] + toothbox[1])/2])
    #print('box_center' ,box_center)
    tooth_center = np.asarray(lm.get_tooth_center(points))
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
    hfactor = (bheight/(hmax - hmin))*0.9
    wfactor = bwidth/(wmax - wmin)
    #print('hfactor and wfactor', hfactor, wfactor)
    centroid = lm.get_tooth_centroid(points)
    #print('centroid', centroid)
    ypoints = points[:,1]
    ypoints = (ypoints - centroid[1]).dot(hfactor) + centroid[1]  
    xpoints = points[:, 0]
    xpoints = (xpoints - centroid[0]).dot(wfactor) + centroid[0] 
    #print('ypoints', ypoints)
    #print('xpoints', xpoints)
    points = np.stack((xpoints, ypoints), axis=-1)
        
    tooth_center = np.asarray(lm.get_tooth_center(points))
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
    cv2.destroyAllWindows()
    
    newpoints = iterate(points, edge_img, img)
    
    return img, newpoints
    
def iterate(points, edge_img, pre_img):
    new_points = points
    #print(points)
    for i in range(len(points)):
        newpoint = get_max_along_normal(points, i, edge_img, pre_img)
        new_points[i] = newpoint
        #i += 1
    cv2.waitKey(0) 
    return new_points  
        
def get_max_along_normal(points, i, edge_img, radiograph):
    # Get the normal to the point 
    px = points[i][0] 
    py = points[i][1]
    
    
    #edge_img = edge_pic
    normal = get_normal_to_point(points, i)
    (ih, iw) = edge_img.shape
   
#    # Print the normal line edges on the Image 
    img = radiograph.copy()
    #print('img.shape', img.shape, 'edge_img.shape', edge_img.shape)
    max_pt = (px, py)
    max_edge = 0     
    
    search = 6
    cv2.circle(img, (int(px), int(py)), 1, (0, 0, 255), 2)
    
    #for t in drange(-search if -search > minR else minR, search if search < maxR else maxR , .5):
    for t in drange(-search, search, 0.5):
        x = int(normal[0]*t + px)
        y = int(normal[1]*t + py
        
        )
        #print('iw, ih, t', iw, ih, t)
        #print('normal[0], normal[1]', normal[0], normal[1])
        if x < 1 or x >= iw - 1  or y < 1 or y >= iw - 1:
            continue
        cv2.circle(img, (int(x), int(y)), 1, (150, 150, 150), 2)
        #print('x, y, edge_img(x, y) ', x, y, edge_img[y, x])
        #print('1.15*maxedge ', 15 + max_edge)
        
        average_edge1 = float((edge_img[y, x] + edge_img[y, (x+1)] + edge_img[y, (x-1)]) / 3)
        average_edge2 = float((edge_img[(y+1), x] + edge_img[(y+1), (x+1)] + edge_img[(y+1), (x-1)]) / 3)  
        average_edge3 = float((edge_img[(y-1), x] + edge_img[(y-1), (x-1)] + edge_img[(y-1), (x+1)]) / 3)      
        average_edge = (average_edge1 + average_edge2 + average_edge3) / 3
        if average_edge > 1.15*max_edge:
            cv2.circle(edge_img, (int(x), int(y)), 3, (0, 0, 250), 3)
            max_edge = edge_img[y, x]
            max_pt = (x, y)
            
    # Print the final point of the model 
    cv2.circle(img, (int(max_pt[0]), int(max_pt[1])), 2, (0, 100, 255), 2)
    #cv2.circle(edge_img, (int(px), int(py)), 2, (200, 200, 200), 2)
    ''' DEcomment next line it if you want to see the intermediate steps ''' 
    # cv2.imshow('Img', img)
    #cv2.imshow('Edge_img', edge_img)
    #cv2.waitKey(0) 
    cv2.destroyAllWindows()   
    
    return np.asarray(max_pt)

def smooth_model(points):
    for i in range(len(points)-1):
        if i == 0:
            points[i][0] = (points[i][0] + points[i+1][0]) / 2
            points[i][1] = (points[i][1] + points[i+1][1]) / 2
        elif i == len(points)-1: 
            points[i][0] = (points[i][0] + points[i-1][0]) / 2
            points[i][1] = (points[i][1] + points[i-1][1]) / 2
        else:
            points[i][0] = (points[i][0] + points[i+1][0] +  points[i-1][0]) / 3
            points[i][1] = (points[i][1] + points[i+1][1] +  points[i-1][1]) / 3
    return points
        
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
        
def get_normal_to_point(points, i):
    dx = 0
    dy = 0
    m = 0
    if i == 0: #first point
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
    elif i == len(points)-1: #last point
        dx = points[-1][0] - points[-2][0]
        dy = points[-1][1] - points[-2][1]
    else:
        dx = points[i+1][0] - points[i-1][0]
        dy = points[i+1][1] - points[i-1][1]
    m = math.sqrt(dx**2 + dy**2)
    
    return (-dy/m, dx/m) 