#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:02:22 2019

@author: yueliair
"""

"""localization.py - localize the blinks in the data file by B-spline wavelet 
transform, thresholding, and watershed 
Some ideas are taken from the paper: Izeddin, I., et al. "Wavelet analysis for 
single molecule localization microscopy." Optics express 20.3 (2012): 2081-2095.
This module detect the circular diffraction limited spots (blinks/smudges) from
single molecule excitation and fit a bounding box around the blinks.
The size of the bounding box is 6x6 pixels, and output the left top corner of 
the bounding box.
All rights reserved.

"""

import tifffile as tiff
import numpy as np
from astropy.convolution import convolve
import scipy
import scipy.ndimage
import cv2
import tkinter
import tkinter.filedialog
import json

# In[]
def ImageFiltering(image):
    
    """
    Filter the image using discrete wavelet transform using B-spline funcitons
    of scale 2 and order 3
    
    Parameters
    ----------
    image: ndarray (2-D) of integers of the raw experimental data
   
    Returns
    -------
    F1: ndarray
        Filtered image, which is the decompostion on the first wavelet plane
    F2: ndarray
        Filtered image, which is the decompostion on the second wavelet plane
        
    Notes
    -----
    This function implements a wavelet decompositio algorithm [1] that remove 
    the noise of the raw data, and provide a clean image for future 
    segmentation.
    In the algorithm, the B-spline function has scale of 2 and order of 3, as 
    suggested in the paper [1]
    
    References
    ----------
    .. [1] Izeddin, I., et al. "Wavelet analysis for single molecule 
    localization microscopy." 
    """
    
    # define kernel1 K1 = k1k1.T for the first wavelet plane
    k1 = np.array([1/16, 1/4,3/8,1/4,1/16]).reshape(5,1)
    K1 = k1*k1.T
    
    # original image convole with kernal K1 to get the first wavelet plane 
    # transform V1 
    V1 = convolve(image, K1)
   
    # define kernel K2 = k2k2.T for the second wavelet plane
    k2 = np.array([1/16, 0, 1/4, 0, 3/8, 0, 1/4, 0, 1/16]).reshape(9,1)
    K2 = k2*k2.T
    
    # Get the filtered image on the first wavelet plane F1 = im-V1
    F1 = image-V1
    
    # find the second wavelet tranform V2 by convolving V1 and K2
    V2 = convolve(V1,K2)

    # calulate the filtered image on the second wavelet plane F2 = V1-V2
    F2 = V1-V2

    return F1, F2

# In[]
def DetectLocalMaxima(F1,F2,thresh = 1.5):
    
    """
    Pass through each pixel of the fitlered image F2 and determines if the 
    pixel value F(x,y) is greater than a user chosen threshold 
    (threshold*std(F1), std(F1) is the estmate of the Guassian noise of the 
    experimental data), and at the same time, the greatest value within the 8-
    connect neighborhood. If this condition holds, then the pixel is selected
    to be a potential molecule for further processing
    
    Parameters
    ----------
    F1: ndarray
        Filtered image, which is the decompostion on the first wavelet plane
    F2: ndarray
        Filtered image, which is the decompostion on the second wavelet plane
    thresh: float
            how many times the pixel needs to be brighter than the background
            Guassian noise. The default value is 1.5
       
    Returns
    -------
    centroid: ndarray
        mask of selected pixels that are potentially molecules
        
    Notes
    -----
    This function detectst the local maximum in 8-connected neighborhood above
    a user defined threshold. The location of the centroid will be further 
    processed as the seed for watershed algorithm in determining the 
    blink/smudge shape
    """
    
   # calculate the threshold thresh
    thresh = thresh*np.std(F1)
    
    # filter the image maxfiltered with max filter with window size of 3x3
    maxfiltered = scipy.ndimage.filters.maximum_filter(F2, size= [3,3])

    # find pixels in the maxfiltered image  with values larger than the 
    # threshold and save the mask as above thresh
    abovethresh = maxfiltered>=thresh
    
    # find pixels in the F2 image with the highest value in the 8 connected 
    # neighthood and save the mask as max8neighbor
    max8neighbor = F2 == maxfiltered
    
    # find the pixels (centroidcandidate) that satisfies both criteria (the 
    # common sets of pixels that are true in both abovethresh and max8neighbor)
    centroidcandidate =  np.multiply(abovethresh,max8neighbor)
    
    # set the edge to be all false, the edge values are artifacts due to 
    #limited image size
    centroid = np.zeros(centroidcandidate.shape, dtype=bool)
    centroid[2:centroid.shape[0]-2,2:centroid.shape[1]-2] = centroidcandidate[2:centroid.shape[0]-2,2:centroid.shape[1]-2]
      
    return centroid

# In[]
def SeparateBlinks(F1,F2,centroid,thresh = 1.5):  
    
    """
    Using watershed algorithm to separate individual blinks with touching
    boundary
    The mask sure_bg is F2 thresholded  by a user chosen value times the 
    Guassian noise level, which can be calcualted by taking the standard 
    deviation of F1, the decomposed image on the first wavelet plane 
    The sure_bg is then dilated to create sure background for watershed 
    The seed of watershed is from the centroid detected from F2
    The separted touching blinks are candidates for additonal process
    
    
    Parameters
    ----------
    F1: ndarray
        Filtered image, which is the decompostion on the first wavelet plane
    F2: ndarray
        Filtered image, which is the decompostion on the second wavelet plane
    thresh: float number
            How many times the pixel needs to be brighter than the background
            Guassian noise. The default value is 1.5 
    centroid:ndarray
             mask of selected pixels that are potentially molecules
    Returns
    -------
    labels: ndarray
            Indexed separted blinks
        
    Notes
    -----
    This function uses the watershed algorihtm to separate touching blinks 
    """
    # calculate the threshold thresh
    thresh = thresh*np.std(F1)
    
    # thresholding F1 and store in T1
    T1 = F2 
    T1[F2<=thresh] = 0
    T1 = T1/T1.max()
    T1 = 255 * T1 # Now scale by 255
    T1 = T1.astype('uint8')
    
     # sure background, sure_bg just right after filtering
    sure_bg = np.zeros(F2.shape, dtype=bool)
    sure_bg[4:T1.shape[0]-4,4:T1.shape[1]-4] = (F2>thresh)[4:T1.shape[0]-4,4:T1.shape[1]-4]
    sure_bg = sure_bg.astype('uint8')*255
    # dilate sure background
    kernel = np.ones((4,4),np.uint8)
    sure_bg = cv2.dilate(sure_bg,kernel,iterations=3)
    
    # sure foreground is the centroid location, convert to uint8 with 255 as 
    # the positive postiion
    sure_fg = centroid.astype('uint8')*255
    
    # Finding unknown region
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    # Threshold F2 to get the mask of all blinks T2
    T2 = (F2>thresh).astype('uint8')*255
    
    # apply watershed to the thresholded F2 image T2, seed is the markers 
    # (thresholded image)
    labels = cv2.watershed(cv2.cvtColor(T2,cv2.COLOR_GRAY2BGR),markers)
    
    return labels
    
# In[]
def CircularBlink(labels,circularity = 0.5,minarea = 2):  
    
    """
    Based on the blinks separated from watershed (labels), determine if the 
    blink is big enough (larger than minarea in pixels) and circular enough 
    (circularity is larger than some user defined threshold).
    Output the x-y coordiate of the top-left corner of the bounding box for 
    blinks that pass both criteria 
    
    Parameters
    ----------
    labels: ndarray
            Indexed separted blinks
    circularity: float
                 user defined threshold of circularity (1 means circle), 
                 deviate from circle means non-circle, default value is 0.6
    minarea: int
             user defined threshold of minimum blink size in pixels, default 
             value is 4 pixels 
   
    Returns
    -------
    x, y: ndarray
          x and y coordinates ofthe top-left corner of the bounding box for 
          blinks that pass both criteria
        
    Notes
    -----
    This function uses circularity to select only circular blinks for output
    """
    # create dynamic list of x and y to store coordinates
    x=[]
    y=[]
    
    # loop through all the separated blinks and filter them by size and 
    # circularity
    for t in range(1,len(np.unique(labels))):
        # select individual blink and store the mask in blink
        blink = np.zeros(labels.shape)
        label = np.unique(labels)[t]
        blink[labels == label]=1
        
        # detect contour of this blink
        contours =cv2.findContours(blink.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        # filter by blink area
        area = cv2.contourArea(contours[1][0])
        
        if area>=minarea:
            
            # find perimeters of the blink
            perimeter = cv2.arcLength(contours[1][0],True)   
            
            # calculate circularity circ based on the ratio of area and 
            # perimeter^2
            circ= (4*np.pi*(area/(perimeter*perimeter))) 
            
            # filter out blinks that are not circular
            if circularity<circ<2-circularity:
                # get centroid, recalculate the position of the molecule using 
                # the moments
                M = cv2.moments(contours[1][0])
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # calculate the top-left corner coordinates
                # the blink size is 6x6
                x.append(cx-3)
                y.append(cy-3)
        
    return x,y  

# In[]
def main(im):
    
    """
    Main function
    locate the circular smudge
    detect the center of the smude
    and output a list of the coordinates of the top-left corner of the 6x6 
    bounding box of the smudge
    
    Parameters
    ----------
    im: ndarray
        experiment data, 3D image stack 
    """
    # initalize dynamic list to store the coordinates of the top-left corner of the bounding bo
    xcor = []
    ycor = []    
       
    # read each frame in the image stack
    for t in range(im.shape[0]):
        
        image = im[t,:,:]
        
        # filtering using DWT, Bpline with scale 2 order 3
        F1,F2 = ImageFiltering(image)
        
        # Finding the centroids
        centroid = DetectLocalMaxima(F1,F2,thresh = 1.5)
        
        # use watershed to separate touching blinks
        labels = SeparateBlinks(F1,F2,centroid,thresh = 1.5)
    
        # select only the circular blinks and calculate the coordinates of the top-left corner of the bounding box
        x,y = CircularBlink(labels,circularity = 0.5,minarea = 2)
        
        # append the coordinates to the previous frame
        xcor.append(x)
        ycor.append(x)
            
    return xcor,ycor

# In[]   
if __name__ == "__main__":
    # ask user to choose file directory
    tkinter.Tk().withdraw() # Close the root window
    in_path = tkinter.filedialog.askopenfilename()
    
    # read the input image from the directory defined by the user
    im = tiff.imread(in_path)
    
    # check the file dimension, if it's one image, add the third dimension to be 1
    if len(im.shape)==2:
        im = im.reshape((1,)+im.shape)
    
    # colocalize the circular blinks and calculate the up-left corner of the bounding box
    xcor, ycor = main(im)
    
    # save the list of the coordinates of all frames
    # open output file for writing
    with open('xcoordinates.txt', 'w') as filehandle:  
        json.dump(xcor, filehandle)
    
    with open('ycoordinates.txt', 'w') as filehandle:  
        json.dump(ycor, filehandle)
    