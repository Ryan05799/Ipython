
# coding: utf-8

# In[73]:

import pandas as pd
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
#%matplotlib inline
import random
from sklearn.lda import LDA
from sklearn.qda import QDA
pd.set_option('display.max_rows', 30)

#Settings
xl_filename = 'feat.xlsx'
feature_x = 'nadh_intensity'
feature_y = 'redox_hetero'
inRedox = True

yaxis_range = 80
xaxis_range = 200
yaxis_range_rdx = 0.15
grid_resol = 255

#Percentage of data being used for training
t_data_perc = 0.334
cv_repeat = 20

#Read cancer data from the excel file
df = pd.read_excel(xl_filename, sheetname = 0, header = 1)
x1 = np.array(df[feature_x]).reshape(-1,1)
y1 = np.array(df[feature_y]).reshape(-1,1)
cancer_pt = np.hstack([x1,y1])

#Read normal data from the excel file
df2 = pd.read_excel('feat.xlsx', sheetname = 1, header = 1)
x2 = np.array(df2[feature_x]).reshape(-1,1)
y2 = np.array(df2[feature_y]).reshape(-1,1)
normal_pt = np.hstack([x2, y2])



# In[74]:

#Sort given training data with corresponding labels
norDataNum,normal_ndim = normal_pt.shape
cnDataNum,cancer_ndim = cancer_pt.shape
nor_n = np.zeros(int(norDataNum))
can_n = np.ones(int(cnDataNum))
labels = np.hstack((nor_n, can_n))
train_data = np.vstack((normal_pt, cancer_pt))


# In[75]:

def randTestData(perc, size):
    
    if perc > 1:
        perc = 1
    
    smpSize = int(perc*size)
    labelTesting = np.ones(size) == 1
    smpIndice = random.sample(list(range(0, size)), smpSize)
    
    for i in smpIndice:
        labelTesting[i] = False
    
    return size-smpSize, labelTesting


# In[76]:

def crossValidate(itr):
    norTrainNum, nor_isTraining = randTestData(t_data_perc, norDataNum)
    cnTrainNum, cn_isTraining = randTestData(t_data_perc, cnDataNum)
    isTraining =np.hstack((nor_isTraining, cn_isTraining))

    #Training
    clf = QDA()
    trained_clf = clf.fit(train_data[isTraining], labels[isTraining])

    #Using the remaining data for testing
    normal_pred = trained_clf.predict(normal_pt[nor_isTraining == False])
    trueneg_n = (normal_pred == 0).sum()
    specificity = trueneg_n/int(norDataNum - norTrainNum)

    cancer_pred = trained_clf.predict(cancer_pt[cn_isTraining == False])
    truepos_n = (cancer_pred == 1).sum()
    sensitivity = truepos_n/int(cnDataNum - cnTrainNum)
    
    #Generate grids for the entire plot
    if inRedox:
        xx, yy = np.meshgrid(np.linspace(0, xaxis_range, grid_resol), np.linspace(0, yaxis_range_rdx , grid_resol))
    else:
        xx, yy = np.meshgrid(np.linspace(0, xaxis_range, grid_resol), np.linspace(0, yaxis_range, grid_resol))

    plot_grid = np.c_[xx.ravel(), yy.ravel()]

    #Calculate the prediction probability for each point on the grid
    grid_z = clf.predict_proba(plot_grid)[:,1].reshape(xx.shape)
    
    
    plt.figure()
    plt.contour(xx, yy, grid_z, [0.5], linewidths=2., colors='k')
    plt.scatter(x1[cn_isTraining == False], y1[cn_isTraining == False], c = 'r', marker = '^', label = 'Cancer (N =' + str(cnDataNum - cnTrainNum) +')')
    plt.scatter(x2[nor_isTraining == False], y2[nor_isTraining == False], c = 'g', marker = '^', label = 'Normal(N =' + str(norDataNum - norTrainNum) +')')

    plt.scatter(x1[cn_isTraining], y1[cn_isTraining], c = 'r', marker = 'o', label = 'Trn_Cancer (N =' + str(cnTrainNum) +')')
    plt.scatter(x2[nor_isTraining], y2[nor_isTraining], c = 'g', marker = 'o', label = 'Trn_Normal(N =' + str(norTrainNum) +')')

    plt.axis('tight')
    plt.xlabel(feature_x, fontsize = 'large')
    plt.ylabel(feature_y, fontsize = 'large')
    plt.legend()
    plt.suptitle(feature_x + ' vs. ' + feature_y, fontsize = 16)
    plt.title('Specificity: ' + '{0:.3f}'.format(specificity) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sensitivity), fontsize = 12)

    plt.savefig('cv' + str(itr) + '.jpg')
    
    return specificity, sensitivity


# In[77]:

result = np.zeros(cv_repeat * 2).reshape((cv_repeat, 2))

for i in range(cv_repeat):
    result[i] = crossValidate(i)


# In[78]:

result[:, 0].sum()/cv_repeat


# In[79]:

result[:, 0].min()


# In[80]:

result[:, 0].max()


# In[81]:

result[:, 1].sum()/cv_repeat


# In[82]:

result[:, 1].min()


# In[83]:

result[:, 1].max()

