# Localisation.py Version 1.0 01/23/2019

## GENERAL USAGE NOTES
- Identifies the areas of light in sample TIFF images (single image and image stack) from a digital microscope
- Detect only circular blinks/smudges
- Output text files contains x (xcoordinates.txt) and y coordinates (ycoordinates.txt) of the top-left corner of the bounding    box of identified blinks/smudges


## PYTHON and PACKAGE REQUIREMENTS
- Python		3.6.4
- tifffile 	0.12.1
- numpy		1.14.2
- astropy 	2.0.2
- scipy		0.19.1
- opencv		3.3.1
- tkinter		8.6.7
- json		2.6.0


## FUNCTIONS
- ImageFiltering: noise removal using B-splines discrete wavelet transform (DWT)
- DetectLocalMaxima: identity centroid of the blinks using thresholding and local-maxima
- SeparateBlinks: separate touching blinks using watershed 
- CircularBlink: identify circular blinks and filter out high intensity single pixels
- main: localize all circular blinks and output text files containing the x and y coordinates of the top-left corner of the bounding box (6x6) for identified blinks


## HOW TO USE
In the command window, type:
$ python Localisation.py


## CONTACT
yueli2014@u.northwestern.edu



 
