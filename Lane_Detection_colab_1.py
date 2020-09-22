# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 07:42:04 2020
updated on July 10
@author: KGaurav
Title: Real Time Road Lane Detection

minLineLength = 100
maxLineGap = 140
"""
import numpy as np
import cv2
import os
import glob

cv2.destroyAllWindows()

def clipVideo(image):
    img = []
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.0, rows * 0.540]   # 1175 (width, height)
    top_left     = [cols * 0.0, rows * 0.400]   # 870
    bottom_right = [cols * 1.0, rows * 0.597]   # 1290
    top_right    = [cols * 1.0, rows * 0.537]   # 1160
    #vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    #cv2.fillPoly(mask, vertices, ignore_mask_color)
    img = image[int(rows * 0.400):int(rows * 0.60), ] # image[height_range, width_range]
    #print(img.shape)
    return img

def RGB_color_selection(image):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    #White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    #Yellow color mask
    #lower_threshold = np.uint8([175, 175,   0])
    #upper_threshold = np.uint8([255, 255, 255])
    #yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    
    #Combine white and yellow masks
    #mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = white_mask)
    
    return masked_image

def convert_hsv(image):
    """
    Convert RGB images to HSV.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def HSV_color_selection(image):
    """
    Apply color selection to the HSV images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    #Convert the input image to HSV
    converted_image = convert_hsv(image)
    
    #White color mask
    lower_threshold = np.uint8([0, 0, 210])
    upper_threshold = np.uint8([255, 30, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    #Combine white and yellow masks
    #mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = white_mask)
    
    return masked_image

def convert_hsl(image):

    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def HSL_color_selection(image):
    """
    Apply color selection to the HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    #Convert the input image to HSL
    converted_image = convert_hsl(image)
    
    #White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    #Combine white and yellow masks
    #mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = white_mask)
    
    return masked_image

def gray_scale(image):

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_smoothing(image, kernel_size = 13):
    """
    Apply Gaussian filter to the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            kernel_size (Default = 13): The size of the Gaussian kernel will affect the performance of the detector.
            It must be an odd number (3, 5, 7, ...).
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold = 50, high_threshold = 150):
    """
    Apply Canny Edge Detection algorithm to the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            low_threshold (Default = 50).
            high_threshold (Default = 150).
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
    image size = 4096 x 2160
    clipped: 950
    """
    mask = np.zeros_like(image)   
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    #print(rows, cols, image.shape)
    top_left     = [cols * 0.0, rows * 0.254]   # 950
    bottom_left  = [cols * 0.0, rows * 0.486]   # 1075 (width, height)
    bottom_right = [cols * 1.0, rows * 0.787]   # 1190
    top_right    = [cols * 1.0, rows * 0.510]   # 1060
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: The output of a Canny transform.
    """
    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 20       #Only lines that are greater than threshold will be returned.
    minLineLength = 50   #Line segments shorter than that are rejected.
    maxLineGap = 125     #Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def draw_lines(image, lines, color = [255, 0, 0], thickness = 2):
    """
    Draw lines onto the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            lines: The lines we want to draw.
            color (Default = red): Line color.
            thickness (Default = 2): Line thickness.
    """
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image



def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    #print("length of points before=",len(lines))
    idx = -1
    for line in lines:
        #print(line)
        for x1, y1, x2, y2 in line:
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if y1 > y2 or length > 150 or x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            if slope < 0 or slope == 0:
                continue
                #left_lines.append((slope, intercept))
                #left_weights.append((length))
            else:
                idx += 1
                right_lines.append([x1,y1,x2,y2])
                #print(right_lines)
                if  idx > 1:
                    if abs(right_lines[idx][0] - right_lines[idx-1][0]) < 10:
                        right_lines.pop(idx)
                        idx -= 1
                #print("intercept=",int(intercept),"slope= %.3f" % slope,"length=",int(length),([x1,y1],[x2,y2]) )
    #left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    #right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    #print("Length of lane=",len(right_lines), right_lines)
    #print("length of points after=",len(right_lines))

    return right_lines

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):

    right_lane = average_slope_intercept(lines)

    return right_lane

    
def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=6):

    line_image = np.zeros_like(image)
    if lines is not None: 
        '''for [x1, y1, x2, y2] in lines: 
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)   '''
      
        for idx, line in enumerate(lines): 
            cv2.line(line_image, (lines[idx][0], lines[idx][1]), (lines[idx][2], lines[idx][3]), color, thickness) 
            if idx > 0:
                cv2.line(line_image, (lines[idx-1][2], lines[idx-1][3]), (lines[idx][2], lines[idx][3]), color, thickness) 
            
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
                 

if __name__ == '__main__':

    # Path of dataset directory 
    path = '/content/drive/My Drive/Colab Notebooks/Colab Notebooks/Videos/1stDrone.MP4'
    #path = "E:\CodeVideoData\ClipDroneData\road_lane.mp4"
    cap = cv2.VideoCapture(path)  
    print("Video Status: ",cap.isOpened())
    
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    frame_count = 0
    dist_dict = {}
    frame_count = 0
    #width = 1500; height = 820
    #width = 3950; height = 2160
    width = 4096; height = 432
    frate = int(cap.get(cv2.CAP_PROP_FPS))
    #frate = 10
    #***************************************************************************************************************
    path21 = '/content/drive/My Drive/Colab Notebooks/Videos/_LaneLines_03.mp4' # Video write
    writer = cv2.VideoWriter(path21, cv2.VideoWriter_fourcc(*'XVID'),frate, (width, height))
    print("Running...")
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame_count += 1
        if frame_count == 3600: # 3600
          break
        clip         = clipVideo(frame)
        color_select = HSL_color_selection(clip)
        gray         = gray_scale(color_select)
        smooth       = gaussian_smoothing(gray)
        edges        = canny_detector(smooth)
        region       = region_selection(edges)
        hough        = hough_transform(region)
        hough.sort(axis=0) 
        #print("Number of points=",len(hough))
        result       = draw_lane_lines(clip, lane_lines(clip, hough))
        
        cv2.imwrite('houghlines_result.jpg',result)
        writer.write(result)
        #cv2.imshow("output", result)
        #cv2.imwrite("road_combo_image.jpg",combo_image)
          
        # When the below two will be true and will press the 'q' on 
        # our keyboard, we will break out from the loop 
        
        # # wait 0 will wait for infinitely between each frames.  
        # 1ms will wait for the specified time only between each frames 

    
        writer.write(result)

    print("finished Successfully")
    print(result.shape)
    cap.release()
    writer.release()
    cv2.destroyAllWindows() 