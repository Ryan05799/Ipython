
# coding: utf-8

# In[25]:

import os
import pandas as pd
from scipy import linalg
from sklearn.qda import QDA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import random


# In[26]:

class QDAResultGen:
    #Attribute members
    #valid
    #dirPath
    #df_nor, df_cn 
    #dataID_nor, dataID_cn
    #dataNum_nor,dataNum_cn
    
    #static attributes
    yaxis_range = 80
    xaxis_range = 255
    yaxis_range_rdx = 0.15
    grid_resol = 255 
    
    #Member Methods 
    #__init__()
    #loadFeatureArr()
    #QDA2DTrain()
    #QDATest()
    #randTrainData()
    #plotQda2dFig()
    
    
    #Constructor
    def __init__(self, resultPath):
        
        self.valid = False
        #check file exist
        if not os.path.exists(resultPath):
            print('Invalid directory path')
            return None
        else:
            try:
                #load excel file to dataframe
                self.dirPath = resultPath
                self.df_nor = pd.read_excel( self.dirPath + 'dataTable.xlsx', sheetname = 1, header = 0)
                self.df_cn = pd.read_excel( self.dirPath + 'dataTable.xlsx', sheetname = 0, header = 0)
            except Exception as e:
                print(e)
                return None
         
        self.dataID_nor = np.array(self.df_nor['0_data ID']).reshape(-1,1)
        self.dataNum_nor = self.dataID_nor.size
        self.dataID_cn = np.array(self.df_cn['0_data ID']).reshape(-1,1)
        self.dataNum_cn = self.dataID_cn.size
                
        self.valid = True
    
    
    #Method for loading required features of all data
    #Two arrays containing the coordinates of data in each class(normal and cancer) will be return
    def loadFeatureArr(self, *features):
        
        if not self.valid:
            print('No data has been loaded')
            return None
        
        #dimension, i.e. the number of features used for analysis
        dim = len(features)
        
        #Format normal data
        f_nor = np.array(self.df_nor[features[0]]).reshape(-1, 1)
        for d in range(1, dim):
            cur_feature = np.array(self.df_nor[features[d]]).reshape(-1, 1)
            f_nor = np.hstack([f_nor, cur_feature])
        
        #Format cancer data
        f_cn = np.array(self.df_cn[features[0]]).reshape(-1, 1)
        for d in range(1, dim):
            cur_feature = np.array(self.df_cn[features[d]]).reshape(-1, 1)
            f_cn = np.hstack([f_cn, cur_feature])
            
        return f_nor, f_cn
    
    
    #Method for training QDA classifier
    #this will train a QDA classifier with given features of the training data 
    #The trained classifier will be returned along with the inside test specificity and sensitivity
    def QDATrain(self, feature_nor, feature_cn):
        
        if not self.valid:
            print('No data has been loaded')
            return None
                
        nor_num, nor_dim = feature_nor.shape
        cn_num, cn_dim = feature_cn.shape
        
        #Format label for each data, 1 stands for cancer
        labels = np.hstack((np.zeros(int(nor_num)), np.ones(int(cn_num))))
        train_data = np.vstack((feature_nor, feature_cn))
        
        #train the QDA classifier
        clf = QDA()
        trained_clf = clf.fit(train_data, labels)
        
        #calculate specificity and sensitivity
        normal_pred = trained_clf.predict(feature_nor)
        specificity = (normal_pred == 0).sum()/nor_num
        
        cancer_pred = trained_clf.predict(feature_cn)
        sensitivity = (cancer_pred == 1).sum()/cn_num
        
        return trained_clf, sensitivity, specificity
    
    #Method for testing the classifier with given testing data
    #Inside test sensitivity and specificity will be returned
    def QDATest(self, clf, test_nor, test_cn):
        
        nor_num, nor_dim = test_nor.shape
        cn_num, cn_dim = test_cn.shape
        
        normal_pred = clf.predict(test_nor)
        trueNeg_nor = (normal_pred == 0).sum()
        specificity = trueNeg_nor/int(nor_num)
        
        cancer_pred = clf.predict(test_cn)
        truePos_cn = (cancer_pred == 1).sum()
        sensitivity = truePos_cn/int(cn_num)
        
        return sensitivity, specificity
    
    #Method for generating mask to mark out data for testing(or training)
    def randTrainData(self, perc, size):
    
        if perc > 1:
            perc = 1

        smpSize = int(perc*size)
        labelTrain = np.ones(size) == 1
        smpIndice = random.sample(list(range(0, size)), smpSize)

        for i in smpIndice:
            labelTrain[i] = False

        return labelTrain    
    
    #Method for ploting the trained QDA classifier and marking all data
    #The X axis and y axis scale and titles must be given
    #The mask (formed in tuple) is used to mark  
    #the result figure will be saved in the result directory
    #assign the string figDir will create 
    def plotQda2dFig(self, clf, normal, cancer, x_scale , y_scale , x_title, y_title , showfig = False, 
                     mask = None, title ='', figDir=''):
        
        #Generate a 2D grid for the entire plot
        xx, yy = np.meshgrid(np.linspace(0, x_scale, self.grid_resol), np.linspace(0, y_scale, self.grid_resol))
        plot_grid = np.c_[xx.ravel(), yy.ravel()]
        #Calculate the prediction probability for each point on the grid
        grid_z = clf.predict_proba(plot_grid)[:,1].reshape(xx.shape)
        plt.figure()
        plt.contour(xx, yy, grid_z, [0.5], linewidths=2., colors='k')
        
        #mark both training and testing data separately if a mask is given
        if mask != None:
            
            plt.scatter(normal[:, 0][mask[0]], normal[:, 1][mask[0]], c = 'b', marker = 'o', 
                        label = 'Normal_trn(N =' + str(normal[mask[0]].size/2) +')')            
            plt.scatter(cancer[:, 0][mask[1]], cancer[:, 1][mask[1]], c = 'r', marker = 'o', 
                        label = 'Cancer_trn (N =' + str(cancer[mask[1]].size/2) +')')
            plt.scatter(normal[:, 0][mask[0] == False], normal[:, 1][mask[0] == False], c = 'b', marker = '^', 
                        label = 'Normal_test(N =' + str(normal[mask[0] == False].size/2) +')')            
            plt.scatter(cancer[:, 0][mask[1] == False], cancer[:, 1][mask[1] == False], c = 'r', marker = '^', 
                        label = 'Cancer_test (N =' + str(cancer[mask[1] == False].size/2) +')')
        else:
            
            plt.scatter(cancer[:, 0], cancer[:, 1], c = 'r', marker = '^', label = 'Cancer (N =' + str(cancer.size/2) +')')
            plt.scatter(normal[:, 0], normal[:, 1], c = 'b', marker = 'o', label = 'Normal(N =' + str(normal.size/2) +')')
            
        plt.axis('tight')
        plt.xlabel(x_title, fontsize = 'large')
        plt.ylabel(y_title, fontsize = 'large')
        plt.legend()
            
        plt.suptitle( x_title + '_vs._' + y_title, fontsize = 16)
        plt.title(title, fontsize = 12)
        
        #create a directory for this plot figure
        if len(figDir) > 0:
            if not os.path.exists(self.dirPath + figDir):
                os.makedirs(self.dirPath + figDir)
            figDir = figDir + '/'
        
        
        #Increase the number if the file already exists
        count = 1
        while os.path.isfile(self.dirPath + figDir + x_title + '_vs._' + y_title + str(count) + '.jpg'):
            count+=1
            
        plt.savefig(self.dirPath + figDir + x_title + '_vs._' + y_title + str(count) + '.jpg')
        
        if showfig:
            plt.show()   
            
        else:
            plt.clf()
            plt.close()
            
        return
            
    #Method for plot 3D QDA result
    #set the parameter boundary for showing the boundary surface 
    def plotQda3dFig(self, clf, normal, cancer, boundary = True, title = ''):
        
        #check data dimesion correct
        if normal.shape[1] != 3 or cancer.shape[1] != 3:
            print('Error: more or less than 3 feature are used for analysis')
            return
            
        #plot the data distribution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(normal[:, 0], normal[:, 1], normal[:, 2], c = 'b', marker = 'o', 
                   label = 'Normal(N =' + str(normal.size/3) +')')
        ax.scatter(cancer[:, 0], cancer[:, 1], cancer[:, 2], c = 'r', marker = '^',
                   label = 'Cancer(N =' + str(cancer.size/3) +')')                 
        
        #show boundary
        if boundary :
            #Generate grids for the entire plot
            xx, yy, zz = np.meshgrid(np.linspace(0, 255, 100), np.linspace(0, 255, 100), np.linspace(0, 0.2, 200))
            plot_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

            #Calculate the prediction probability for each point on the grid
            grid_result = clf.predict_proba(plot_grid)[:,1].reshape(xx.shape)
            a = abs(grid_result - 0.5)
            sur_x, sur_y = np.meshgrid(np.linspace(0, 255, 100), np.linspace(0, 255, 100))
            sur_z = np.zeros(sur_x.size).reshape(sur_x.shape)

            for i in range(100):
                for j in range(100):     
                    sur_z[i][j] = zz[i][j][a[i][j].argmin()]
                
            ax.plot_wireframe(sur_x, sur_y, sur_z,  rstride=10, cstride=10, color='g')
                            
        plt.title(title, fontsize = 12)
        plt.suptitle('FAD intensity - NADH intensity - Heterogeneit redox', fontsize = 14)
        ax.set_xlabel('FAD_intensity')
        ax.set_ylabel('NADH_intensity')
        ax.set_zlabel('redox_hetero')
        plt.legend()
        plt.show()
        plt.clf()
        return
        
    #This method does the inside test for given data,
    #the process will be repeat in given number of iterations, 
    #each time with the given percentage of data as testing data
    
    def crossValidate(self, itr, *features, perc = 0.333, savefig = True):
        
        normal, cancer = self.loadFeatureArr(*features)
        
        #array for saving results(sensitivity and specificity)
        result = np.zeros(itr*2).reshape((itr, 2))

        
        dirName = ' '.join(['CV_',str(features), str(itr), 'iter'])
        if not os.path.exists(self.dirPath + dirName):
            os.makedirs(self.dirPath + dirName)

        for i in range(itr):
            #gererate mask to label training and testing data
            mask = (self.randTrainData(perc, self.dataNum_nor), self.randTrainData(perc, self.dataNum_cn))
            trn_normal = normal[mask[0]]
            trn_cancer = cancer[mask[1]]
            test_normal = normal[mask[0] == False]
            test_cancer = cancer[mask[1] == False]
            clf, sp, sn = self.QDATrain(trn_normal, trn_cancer)
            result[i] = self.QDATest(clf, test_normal, test_cancer)
            
            #Save result plot if savefig is set and only 2 features are used
            if (savefig and len(features) == 2):
                tag = 'Outside test Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
                if ('redox' in features[1]):
                    y_range = 0.15
                else:
                    y_range = 80
                    
                self.plotQda2dFig(clf, normal, cancer, 255, y_range, features[0], features[1]
                                    , showfig = False, mask = mask, title = tag, figDir = dirName)
        
        #Plot 3D figure if 3 features are used
        if(len(features) == 3):
            tag = 'Outside test Specificity: ' + '{0:.3f}'.format(result[:, 1].sum()/itr) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(result[:, 0].sum()/itr)
            self.plotQda3dFig(None, normal, cancer, boundary = False, title = tag)
        
        #save the result statistics to file
        f = open(self.dirPath + dirName +'/log.txt', 'w') 
        f.write( '0_Repeat : ' + str(itr) + '\n')
        f.write( '1_Training data percentage : ' + str(perc) + '\n')
        f.write( '2_features : ' + str(features) + '\n')
        f.write( '3_Sensitivity_Results : ' + str(result[:, 0]) + '\n')
        f.write( '4_Specificity_Results : ' + str(result[:, 1]) + '\n')
        f.write( '5_Sensitivity_avr : ' + str(result[:, 0].sum()/itr) + '\n')
        f.write( '6_Sensitivity_min : ' + str(result[:, 0].min()) + '\n')
        f.write( '7_Sensitivity_max: ' + str(result[:, 0].max()) + '\n')
        f.write( '8_specificity_avr : ' + str(result[:, 1].sum()/itr) + '\n')
        f.write( '9_specificity_min : ' + str(result[:, 1].min()) + '\n')
        f.write( '10_specificity_max : ' + str(result[:, 1].max()) + '\n')        
        f.close()
            
        return result
    
    #weaklearner
    def weakLearner(self, normal, cancer, test_data, perc):
        
        predict = np.zeros(test_data.shape[0])
        
        for i in range(test_data.shape[0]):
            votes = np.zeros(5)
            for v in range(5):
                #take half of the given data for training
                mask = (self.randTrainData(perc, normal.shape[0]), self.randTrainData(perc, cancer.shape[0]))
                trn_normal = normal[mask[0]]
                trn_cancer = cancer[mask[1]]
                clfs, sp, sn = self.QDATrain(trn_normal, trn_cancer)
                votes[v] = clfs.predict(test_data[i])
#             print( 'data ' + str(i) + ' ' + str(votes) + '\n')
            predict[i] = (votes.sum() > 2)
        
        return predict

    #This method load the given data features for Weak learner classifying
    #The assigned percentage of data will be used for testing
    #The remaining data will be used for building weak learners
    #Argument itr sets the number of iterations
    def crossValidateWL(self, itr, *features, perc = 0.5, perc_trn = 0.5, saveResult = False):
        #array for saving results(sensitivity and specificity)
        print('Weak Learner cross validation using features: ' + str(features) + '\nfor ' + str(itr) + 'iter')
        normal, cancer = self.loadFeatureArr(*features)
        result = np.zeros(itr*2).reshape((itr, 2))
        for i in range(itr):
            #gererate mask to label training and testing data
            mask = (self.randTrainData(perc, self.dataNum_nor), self.randTrainData(perc, self.dataNum_cn))
            test_normal = normal[mask[0] == False]
            test_cancer = cancer[mask[1] == False]
            
#             print('Testing normal data')
            trueNeg = (self.weakLearner(normal[mask[0]], cancer[mask[1]], test_normal, 1 - perc_trn) == 0).sum() 
            
#             print('Testing cancer data')
            truePos = (self.weakLearner(normal[mask[0]], cancer[mask[1]], test_cancer, 1 - perc_trn) == 1).sum() 
            
            result[i] = (truePos/ test_cancer.shape[0]), (trueNeg/ test_normal.shape[0])
        
        #save the result statistics to file
        dirName = ' '.join(['WeakLernerCV_', str(features),str(itr), 'iter'])

        if not os.path.exists(self.dirPath + dirName):
            os.makedirs(self.dirPath + dirName)
            
        f = open(self.dirPath + dirName +'/log.txt', 'w') 
        f.write( '0_Repeat : ' + str(itr) + '\n')
        f.write( '1_Training data percentage : ' + str(perc) + '\n')
        f.write( '2_Features : ' + str(features) + '\n')
        f.write( '3_Sensitivity_Results : ' + str(result[:, 0]) + '\n')
        f.write( '4_Specificity_Results : ' + str(result[:, 1]) + '\n')
        f.write( '5_Sensitivity_avr : ' + str(result[:, 0].sum()/itr) + '\n')
        f.write( '6_Sensitivity_min : ' + str(result[:, 0].min()) + '\n')
        f.write( '7_Sensitivity_max: ' + str(result[:, 0].max()) + '\n')
        f.write( '8_specificity_avr : ' + str(result[:, 1].sum()/itr) + '\n')
        f.write( '9_specificity_min : ' + str(result[:, 1].min()) + '\n')
        f.write( '10_specificity_max : ' + str(result[:, 1].max()) + '\n')        
        f.close()
        
        return result            

