# -*- coding: utf-8 -*-
import cv2.cv as cv
import cv2
import numpy as np
import os

box_coordinates = []
condition = True
point1 = (0, 0)
point2 = (0, 0)
point2tmp = (0, 0)
drawing = False

#def get_box_per_tooth(radiograph, index):
#    # grab references to the global variables
#    global point1, point2, box_coordinates, condition
#    
#    # Draw the box around one tooth, it's sayi wich one.
#    if index < 5:
#        window_title = 'Draw a box around the upper incisor number ' + str(index) + ' from the left' 
#    else:
#        index = index % 4 
#        window_title = 'Draw a box around the lower incisor number ' + str(index) + ' from the left' 
#    
#    # Setup the mouse callback function
#    cv2.namedWindow(window_title)
#    cv2.setMouseCallback(window_title, mouse_callback_function)
#    
#    while condition:
#        # show the image, with the rectangle if present 
#        rect_cpy = radiograph.copy()
#        if point1 != (0, 0):
#            cv2.rectangle(rect_cpy, point1, point1, (255, 0, 0), 1)
#            cv2.imshow(window_title, rect_cpy)
#        elif point1 != (0, 0) and point2 != (0, 0):
#            cv2.rectangle(rect_cpy, point1, point2, (255, 0, 0), 1)
#            cv2.imshow(window_title, rect_cpy)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
#            break
#        elif point1 == (0,0): 
#            cv2.imshow(window_title, radiograph)
#        key = cv2.waitKey(1) & 0xFF
#        if key == 27: 
#            # 27 is ESC
#            break
#            
#    boxes.append(box_coordinates)
#    print('boxes len', len(boxes))
#    # Reset the variables for the next loop
#    reset_global_variable()
#    cv2.destroyAllWindows() 
#    return boxes

def get_box_per_jaw(radiograph, i, where):
    '''
    Displays the image, and ask the user to print a box over the 'where' jaw.
    Parameters: 
        radiograph; image to show
        i; index of the tooth
        where; says if the box to draw is around upper or lower teeth
    '''
    # grab references to the global variables
    global point1, point2, box_coordinates, condition, point2tmp, drawing
    
    window_title = 'Sample ' + str(i) + ': draw a box around the ' + where + ' teeth' 
    
    # Setup the mouse callback function
    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, mouse_callback_function)
    
    while condition:
        # show the image, with the rectangle if present 
        rect_cpy = radiograph.copy()
        if not drawing:
            cv2.imshow(window_title, radiograph)
        else:
            cv2.rectangle(rect_cpy, point1, point2tmp, (0, 0, 255), 1)
            cv2.imshow(window_title, rect_cpy)
            
        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            # 27 is ESC
            break
    
    box = box_coordinates
    # Reset the variables for the next loop
    reset_global_variable()
    cv2.destroyAllWindows() 
    return box
      
def reset_global_variable():
    '''
    Function to reset the global variables and clean the memory for the 
    next coming calls. Also restart the drawing over the image. 
    '''
    global point1, point2, point2tmp, box_coordinates, condition, drawing
    point1 = (0, 0)
    point2 = (0, 0)
    # THIS ERASES EVERYTHING; KEEP A GLOBAL VARIABLE: 
    box_coordinates = []
    condition = True
            
def mouse_callback_function(ev, x, y, flags, param):
    '''
    Callback for the mouse movements.
    Takes the coordinates of the boxes, when there is a click down the first 
    point is taken, the second one when it is released. Whenever there is 
    a move, store it in point2tmp
    '''
    # grab references to the global variables
    global point1, point2, box_coordinates, condition, point2tmp, drawing 
    
    if ev == cv2.EVENT_LBUTTONDOWN:
        # Button was clicked, store the first coordinate 
        point1 = (x, y)
        point2tmp = (x, y)
        drawing = True
    elif ev == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        drawing = False 

        # Extract the coordinates
        p1x, p1y = point1
        p2x, p2y = point2

        # Take the biggest box possible with the current coordinates
        leftx = min(p1x, p2x)
        topy = min(p1y, p2y)
        rightx = max(p1x, p2x)
        bottomy = max(p1y, p2y)
        
        # add bbox to list if both points are different
        if (leftx, topy) != (rightx, bottomy):
            box_coordinates = [leftx, topy, rightx, bottomy]
        # Break the while loop showing the image
        condition = False
    # if mouse is drawing set tmp rectangle endpoint to (x,y)
    elif ev == cv2.EVENT_MOUSEMOVE and drawing:
        point2tmp = (x, y)
        
def print_boxes_on_tooth(boxes, radiograph):
    ''' 
    Given the boxes and the radiograph, the function prints a box around 
    each tooth. The dimension are obtained dividing the boxes in 4 equal
    parts. 
    An offset to wide or reduce the width of the boxes is applicable.
    ''' 
    # Offset to change the width of the boxes 
    offset = 0
    img = radiograph.copy()
    
    # For the upper boxes 
    lx = boxes[0]
    rx = boxes[2]
    width = rx - lx
    width = int(width / 4)
    width2 = 2 * width
    cv2.rectangle(img, (lx - offset, boxes[1]), (lx + width + offset, boxes[3]), (0, 255, 0), 1)
    cv2.rectangle(img, (lx + width - offset, boxes[1]), (lx + width2 + offset, boxes[3]), (255, 0, 0), 1)
    cv2.rectangle(img, (lx + width2 - offset, boxes[1]), (rx - width + offset, boxes[3]), (0, 0, 255), 1)
    cv2.rectangle(img, (rx - width - offset, boxes[1]), (rx + offset, boxes[3]), (0, 255, 0), 1)
    
    # For the lower boxes 
    lx = boxes[4]
    rx = boxes[6]
    width = rx - lx
    width = int(width / 4)
    width2 = 2 * width
    cv2.rectangle(img, (lx - offset, boxes[5]), (lx + width + offset, boxes[7]), (0, 255, 0), 1)
    cv2.rectangle(img, (lx + width - offset, boxes[5]), (lx + width2 + offset, boxes[7]), (255, 0, 0), 1)
    cv2.rectangle(img, (lx + width2 - offset, boxes[5]), (rx - width + offset, boxes[7]), (0, 0, 255), 1)
    cv2.rectangle(img, (rx - width - offset, boxes[5]), (rx + offset, boxes[7]), (0, 255, 0), 1)
    
    # Show the image
    cv2.imshow('Radiograph with boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def print_boxes_on_teeth(boxes, image):
    '''
    Prints the boxes (not detailed) around the upper and lower jaws.
    Parameters:
        boxes; boxes coordinates
        image; radiograph
    '''
    img = image.copy()
    cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 255, 0), 1)
    cv2.rectangle(img, (int(boxes[4]), int(boxes[5])), (int(boxes[6]), int(boxes[7])), (255, 0, 0), 1)
    cv2.imshow('Radiograph with boxes', img)
    cv2.waitKey(0)
    return
    
def save_boxes(boxes):
    '''
    Given all the boxes, it saves them in a text file, each line a list 
    of the coordinates for the upper and lower jaws. 
    An offset to wide or reduce the width of the boxes is applicable.
    '''
    #directory = '../_Data/Radiographs/'
    #dir_path = os.path.join(os.getcwd(), directory)
    text_file = open("boxes.txt", "w")
    # Offset to change the width of the boxes 
    offs = 0
    
    for i in range(boxes.shape[0]):    
        text_file.write("{} {} {} {} {} {} {} {}\n".format((boxes[i][0] - offs), boxes[i][1], (boxes[i][2] + offs), boxes[i][3],\
                                                            (boxes[i][4] - offs), boxes[i][5], (boxes[i][6] + offs), boxes[i][7]))
    text_file.close()
    return 
    
#def save_boxes_detailed(boxes):
#    #directory = '../_Data/Radiographs/'
#    #dir_path = os.path.join(os.getcwd(), directory)
#    
#    text_file = open("boxes.txt", "w")
#    offs = 10
#    
#    for i in range(boxes.shape[0]):
#        ulx = boxes[i][0]
#        urx = boxes[i][2]
#        uwidth = urx - ulx
#        uwidth = int(uwidth / 4)
#        uwidth2 = 2 * uwidth
#        llx = boxes[i][4]
#        lrx = boxes[i][6]
#        lwidth = lrx - llx
#        lwidth = int(lwidth / 4)
#        lwidth2 = 2 * lwidth
#    
#        text_file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
#            (ulx - offs), boxes[i][1], (ulx + uwidth + offs), boxes[i][3],
#            (ulx + uwidth - offs), boxes[i][1], (ulx + uwidth2 + offs), boxes[i][3],
#            (ulx + uwidth2 - offs), boxes[i][1], (urx - uwidth + offs), boxes[i][3], 
#            (urx - uwidth - offs), boxes[i][1], (urx + offs), boxes[i][3],
#            (llx - offs), boxes[i][5], (llx + lwidth + offs), boxes[i][7], 
#            (llx + lwidth - offs), boxes[i][5], (llx + lwidth2 + offs), boxes[i][7], 
#            (llx + lwidth2 - offs), boxes[i][5], (lrx - lwidth + offs), boxes[i][7], 
#            (lrx - lwidth - offs), boxes[i][5], (lrx + offs), boxes[i][7]             ))
#        
#    text_file.close()

def read_boxes_from_file():
    '''
    Read the stored coordinates of the boxes from the file.
    Returns: the boxes as list of coordinates
    '''
    with open('boxes.txt') as f:
        lines = []
        for line in f:
            lines.append([int(s) for s in line.split() if s.isdigit()])
    lines = np.asarray(lines)
    return lines
    
def get_mean_boxes(boxes):
    '''
    Given all the coordinates of all the boxes, it
    computes and returns the mean box for each jaw. 
    '''
    mean = np.mean(boxes, axis=0)
    return mean 
    
def get_largest_boxes(boxes): 
    '''
    Given all the coordinates of all the boxes, it finds the largest 
    possible boxes for the upper and lower jaws. 
    Parameters:
        Boxes: matrix containg all the boxes for all the samples. 
    '''
    uminx = boxes[:,0].min()
    uminy = boxes[:,1].min()
    umaxx = boxes[:,2].max()
    umaxy = boxes[:,3].max()
    lminx = boxes[:,4].min()
    lminy = boxes[:,5].min()
    lmaxx = boxes[:,6].max()
    lmaxy = boxes[:,7].max()
    
    # Return the largest boxes
    return [uminx, uminy, umaxx, umaxy, lminx, lminy, lmaxx, lmaxy]
    