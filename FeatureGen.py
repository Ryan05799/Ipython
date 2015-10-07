
# coding: utf-8

# In[14]:

import os
import os.path
from skimage import data, io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FeatureGen:
    #members attrigutes
    #     dataID - target data ID, used in forming file path
    #     dirPath - the directory where the target data  is in
    #     valid - If the given dataID and dirPath exists and forming valid file path
    #     singleImage - using only FAD image or both FAD and NADH images
    #     fadImg
    #     nadhImg
    #     maskImg
    #     redoxImgPix 
    #     redoxImgGrid
    
    
    #methods
    
    def __init__(self, id, path, single = False):
        
        self.dataID = str(id)
        self.dirPath = path + str(id)
        self.singleImg = single
        
        #check file path correct
        if not os.path.exists(self.dirPath):
            print('Invalid file path ' + self.dirPath)
            self.valid = False
            return None
        else:
            try:
                self.valid = True
                files = os.listdir(self.dirPath)
            except Exception as e:
                print(e)
        
        if(self.singleImg):
            #Read only single image for analysis
            self.fadImg = self.getImgFile(files, '460nm')
            self.maskImg =  self.getImgFile(files, 'binMask')
                      
            #Exclude the non-ROI part
            self.fadImg[self.maskImg == 0] = 0
            print('Load FAD image')
            
        else:
            #Read a pair of images
            self.fadImg = self.getImgFile(files, '460nm')
            self.nadhImg =  self.getImgFile(files, '375nm')
            self.maskImg =  self.getImgFile(files, 'binMask')
            
            #Exclude the non-ROI part
            self.nadhImg[self.maskImg == 0] = 0
            self.fadImg[self.maskImg == 0] = 0
            print('Load NADH and FAD images')    
    
    def getImgFile(self, filelist, tag):
        for file in filelist:
            if tag in file:
                img = data.imread(self.dirPath + '/'+ file)
                return img       
    
    #Method for calculate the average of RGB or Graylevel in ROI
    def rgbAver(self, chan, imgarr, msk):
        #set chan to 0, 1, 2 for R, G, B channel
        #set chan to -1 for gray level
        avr = 0.0

        if chan < 0:
            grayimg = imgarr[:, :, 0] * 0.299 + imgarr[:, :, 1] * 0.587 + imgarr[:, :, 2] * 0.114
            avr = np.average(grayimg[msk != 0])
            return avr

        elif chan < 3:
            imgChan = imgarr[:, :, chan]
            avr = np.average(imgChan[msk != 0])
            return avr
        
        return None
    
    #Method for calculate the standard deviation of RGB or Graylevel in ROI
    def rgbStd(self, chan, imgarr, msk):
        #set chan to 0, 1, 2 for R, G, B channel
        #set chan to -1 for gray level
        if chan < 0:
            grayimg = imgarr[:, :, 0] * 0.299 + imgarr[:, :, 1] * 0.587 + imgarr[:, :, 2] * 0.114
            std = np.std(grayimg[msk != 0])
            return std

        elif chan < 3:
            imgChan = imgarr[:, :, chan]
            std = np.std(imgChan[msk != 0])
            return std
        return None
    
    #Method for generating the pixel by pixel redox ratio image and its std
    #RR = img1/(img1 + img2)
    def redoxRatioPix(self, img1, img2, mask):
        img_h, img_w = mask.shape
        redoxImg = np.zeros(img_h * img_w).reshape((img_h, img_w))
        for y in range(img_h):
            for x in range(img_w):
                if mask[y, x] != 0:
                    redoxImg[y, x] = float(img1[y, x]) / (float(img1[y, x]) + float(img2[y, x]))

        #Standard Variation of the redox ratio image
        RedoxStd = np.std(redoxImg[mask != 0])
        return redoxImg, RedoxStd
    
    #Generate the grid by grid redox ratio
    #Each grid has the size of 3x3
    #RR = img1/(img1 + img2)
    def redoxRatioGrid(self, imgG1, imgG2, mask):

        #initialize grids to zeros
        grid_h = int(mask.shape[0]/3)
        grid_w = int(mask.shape[1]/3)
        gridNadh = np.zeros(grid_h * grid_w).reshape((grid_h, grid_w))
        gridFad = np.zeros(grid_h * grid_w).reshape((grid_h, grid_w))
        gridRedox = np.zeros(grid_h * grid_w).reshape((grid_h, grid_w))

        redoxImg = np.zeros(mask.size).reshape(mask.shape)

        #Calculate the redox ratio for each grid
        for y in range(grid_h):
            for x in range(grid_w):

                window = mask[3*y:3*y+3, 3*x:3*x+3]
                validPixNum = (window != 0).sum()

                #Only grids that has more than 1 pixel is in ROI will be taken into calculation
                if validPixNum != 0:
                    gridNadh[y][x] = int(imgG1[3*y:3*y+3, 3*x:3*x+3].sum()/validPixNum)
                    gridFad[y][x] = int(imgG2[3*y:3*y+3, 3*x:3*x+3].sum()/validPixNum)
                    gridRedox[y][x] = gridNadh[y][x]/(gridNadh[y][x] + gridFad[y][x])

                    #Map the grids into the original image size
                    redoxImg[3*y:3*y+3, 3*x:3*x+3] = gridRedox[y][x]

        #Standard Variation of the redox ratio image
        ROI = redoxImg[mask != 0]
        RedoxStd = np.std(ROI)

        return redoxImg, RedoxStd
    
    #Method for generating output feature table dict    
    def outputDict(self):

        featureTable = {'0_data ID': self.dataID }
        
        featureTable['Fad_Intensity_R'] = self.rgbAver(0, self.fadImg, self.maskImg)
        featureTable['Fad_Intensity_G'] = self.rgbAver(1, self.fadImg, self.maskImg)
        featureTable['Fad_Intensity_B'] = self.rgbAver(2, self.fadImg, self.maskImg)
        featureTable['Fad_Intensity_Graylevel'] = self.rgbAver(-1, self.fadImg, self.maskImg)
        featureTable['Fad_std_R'] = self.rgbStd(0, self.fadImg, self.maskImg)
        featureTable['Fad_std_G'] = self.rgbStd(1, self.fadImg, self.maskImg)
        featureTable['Fad_std_B'] = self.rgbStd(2, self.fadImg, self.maskImg)
        
        if not self.singleImg:
            featureTable['NADH_Intensity_R'] = self.rgbAver(0, self.nadhImg, self.maskImg)
            featureTable['NADH_Intensity_G'] = self.rgbAver(1, self.nadhImg, self.maskImg)
            featureTable['NADH_Intensity_B'] = self.rgbAver(2, self.nadhImg, self.maskImg)
            featureTable['NADH_Intensity_Graylevel'] = self.rgbAver(-1, self.nadhImg, self.maskImg)
            featureTable['NADH_std_R'] = self.rgbStd(0, self.nadhImg, self.maskImg)
            featureTable['NADH_std_G'] = self.rgbStd(1, self.nadhImg, self.maskImg)
            featureTable['NADH_std_B'] = self.rgbStd(2, self.nadhImg, self.maskImg)
            self.redoxImgPix, featureTable['redoxRatioPix'] = self.redoxRatioPix(self.nadhImg[:, :, 2], self.fadImg[:, :, 1], self.maskImg)
            self.redoxImgGrid, featureTable['redoxRatioGrid'] = self.redoxRatioGrid(self.nadhImg[:, :, 2], self.fadImg[:, :, 1], self.maskImg)

        return featureTable
        
    #Method for output results file
    def outputResultFile(self):    
        writer = pd.ExcelWriter(self.dirPath + '/feature_results.xlsx')
        dfout = pd.DataFrame(self.outputDict(), index = [0])
        dfout.to_excel(writer, 'sheet1')
        writer.save()
        

        
        if not self.singleImg:
            sp1 = plt.subplot(221)
            sp1.set_title('FAD ROI')
            io.imshow(self.fadImg)
            
            sp2 = plt.subplot(222)
            sp2.set_title('NADH ROI')
            io.imshow(self.nadhImg)

            sp3 = plt.subplot(223)
            sp3.set_title('Redox ratio image (pixels)')
            io.imshow(self.redoxImgPix)

            sp4 = plt.subplot(224)
            sp4.set_title('Redox ratio image (Grids)')
            io.imshow(self.redoxImgGrid)
            plt.savefig(self.dirPath + '/redox ratio.jpg')


