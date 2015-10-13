
# coding: utf-8

# In[7]:

import os
import os.path
from skimage import data, io, color
import matplotlib.pyplot as plt
import numpy as np


# In[8]:

#Function for converting RGB to YCbCr
def rgb2ycbcr(img):
    y = 0.299*img[:,:,0] + 0.578*img[:,:,1] + 0.114*img[:,:,2]
    cr = 0.5*img[:,:,0] - 0.4187*img[:,:,1] + 0.0813*img[:,:,2] + 128
    cb = -0.1687*img[:,:,0] - 0.3313*img[:,:,1] + 0.5*img[:,:,2] + 128
    
    return y,cb,cr

#Main function for remove the reflectance of given image
#ROI mask must be given
def rmReflect(img, roiMask, threshold = 0):
    
    #Generate lamda factor    
    yChan, cbChan, crChan = rgb2ycbcr(img)   
    gamma = yChan/(yChan+cbChan+crChan)
    
    maxRgb = np.amax(img, axis = 2)
    #Generate lamda factor
    rgbSum = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    lamb = maxRgb
    lamb[roiMask != 0] = lamb[roiMask != 0]/rgbSum[roiMask != 0]
    
    #Mark the pixels whose luminace values are low than the threshold as 255(white)
    luminace = (gamma - lamb)*255
    mask = np.zeros(luminace.size).reshape(luminace.shape)
    mask[luminace >= threshold ] = 0    
    mask[luminace < threshold] = 255
    
    img[roiMask == 0] = 0
    
    #remove the reflectance regions within the ROI
    roiMask[mask == 255] = 0
    
    return img, roiMask

