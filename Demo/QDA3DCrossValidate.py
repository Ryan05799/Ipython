
# coding: utf-8

# In[107]:

import pandas as pd
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.qda import QDA
import random
pd.set_option('display.max_rows', 30)


# In[108]:

#Settings
t_data_perc = 0.4


# In[109]:

#Read cancer data from the excel file
df = pd.read_excel('feat.xlsx', sheetname = 0, header = 1)
x1 = np.array(df['nadh_intensity']).reshape(-1,1)
y1 = np.array(df['fad_intensity']).reshape(-1,1)
z1 = np.array(df['redox_hetero']).reshape(-1,1)
cancer_pt = np.hstack([x1, y1, z1])

#Read normal data from the excel file
df2 = pd.read_excel('feat.xlsx', sheetname = 1, header = 1)
x2 = np.array(df2['nadh_intensity']).reshape(-1,1)
y2 = np.array(df2['fad_intensity']).reshape(-1,1)
z2 = np.array(df2['redox_hetero']).reshape(-1,1)
normal_pt = np.hstack([x2, y2, z2])

norDataNum, normal_ndim = normal_pt.shape
cnDataNum, cancer_ndim = cancer_pt.shape

#Sort given training data with corresponding labels
nor_n = np.zeros(int(normal_pt.size/normal_ndim))
can_n = np.ones(int(cancer_pt.size/cancer_ndim))
labels = np.hstack((nor_n, can_n))
train_data = np.vstack((normal_pt, cancer_pt))


# In[110]:

def randTestData(perc, size):
    
    if perc > 1:
        perc = 1
    
    smpSize = int(perc*size)
    labelTesting = np.ones(size) == 1
    smpIndice = random.sample(list(range(0, size)), smpSize)
    
    for i in smpIndice:
        labelTesting[i] = False
    
    return size-smpSize, labelTesting


# In[111]:

def QDAResult3D():

    norTrainNum, nor_isTraining = randTestData(t_data_perc, norDataNum)
    cnTrainNum, cn_isTraining = randTestData(t_data_perc, cnDataNum)
    isTraining =np.hstack((nor_isTraining, cn_isTraining))

    #Training QDA classifier
    clf = QDA()
    trained_clf = clf.fit(train_data[isTraining], labels[isTraining])

     #Using the remaining data for testing
    normal_pred = trained_clf.predict(normal_pt[nor_isTraining == False])
    trueneg_n = (normal_pred == 0).sum()
    specificity = trueneg_n/int(norDataNum - norTrainNum)

    cancer_pred = trained_clf.predict(cancer_pt[cn_isTraining == False])
    truepos_n = (cancer_pred == 1).sum()
    sensitivity = truepos_n/int(cnDataNum - cnTrainNum)
    
    return sensitivity, specificity


# In[112]:

result = np.zeros(40).reshape((20, 2))

for i in range(20):
    result[i] = QDAResult3D()

specAvr = result[:, 1].sum()/20
sensAvr = result[:, 0].sum()/20


# In[113]:

#plot the result in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1.ravel(), y1.ravel(), z1.ravel(), c = 'r', marker = 'o', label = 'Cancer(N =' + str(cnDataNum) +')')
ax.scatter(x2.ravel(), y2.ravel(), z2.ravel(), c = 'b', marker = '^', label = 'Normal(N =' + str(norDataNum) +')')

plt.axis('tight')
plt.title('Specificity: ' + '{0:.3f}'.format(specAvr) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sensAvr), fontsize = 12)
plt.suptitle('NADH intensity - FAD intensity - Heterogeneit redox', fontsize = 14)
ax.set_xlabel('nadh_intensity')
ax.set_ylabel('fad_intensity')
ax.set_zlabel('redox_hetero')
plt.legend()
plt.show()

