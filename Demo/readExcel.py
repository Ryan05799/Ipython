
# coding: utf-8

# In[44]:

import pandas as pd
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
#%matplotlib inline

from sklearn.lda import LDA
from sklearn.qda import QDA
pd.set_option('display.max_rows', 30)



# In[45]:

#Settings
xl_filename = 'feat.xlsx'
feature_x = 'nadh_intensity'
feature_y = 'redox_hetero'
inRedox = True

yaxis_range = 255
xaxis_range = 200
yaxis_range_rdx = 0.15
grid_resol = 255


# In[46]:

#Read cancer data from the excel file
df = pd.read_excel(xl_filename, sheetname = 0, header = 1)
x = np.array(df[feature_x])
X = x.reshape(-1,1)
y = np.array(df[feature_y])
Y = y.reshape(-1,1)
cancer_pt = np.hstack([X,Y])


# In[47]:

#Read normal data from the excel file
df2 = pd.read_excel('feat.xlsx', sheetname = 1, header = 1)
x2 = np.array(df2[feature_x]).reshape(-1,1)
y2 = np.array(df2[feature_y]).reshape(-1,1)
normal_pt = np.hstack([x2, y2])


# In[48]:

#Sort given training data with corresponding labels
nor_n = np.zeros(int(normal_pt.size/normal_pt.ndim))
can_n = np.ones(int(cancer_pt.size/cancer_pt.ndim))
labels = np.hstack((nor_n, can_n))
train_data = np.vstack((normal_pt, cancer_pt))


# In[49]:

clf = QDA()
trained_clf = clf.fit(train_data, labels)
normal_pred = trained_clf.predict(normal_pt)
trueneg_n = (normal_pred == 0).sum()
specificity = trueneg_n/int(normal_pt.size/normal_pt.ndim)


# In[50]:

cancer_pred = trained_clf.predict(cancer_pt)
truepos_n = (cancer_pred == 1).sum()
sensitivity = truepos_n/int(cancer_pt.size/cancer_pt.ndim)


# In[51]:

#Generate grids for the entire plot
if inRedox:
    xx, yy = np.meshgrid(np.linspace(0, xaxis_range, grid_resol), np.linspace(0, yaxis_range_rdx , grid_resol))
else:
    xx, yy = np.meshgrid(np.linspace(0, xaxis_range, grid_resol), np.linspace(0, yaxis_range, grid_resol))

plot_grid = np.c_[xx.ravel(), yy.ravel()]

#Calculate the prediction probability for each point on the grid
grid_z = clf.predict_proba(plot_grid)[:,1].reshape(xx.shape)


# In[99]:

xx


# In[95]:

plt.figure()
plt.contour(xx, yy, grid_z, [0.5], linewidths=2., colors='k')

plt.scatter(X, Y, c = 'r', marker = '^', label = 'Cancer (N =' + str(cancer_pt.size/2) +')')
plt.scatter(x2, y2, c = 'b', marker = 'o', label = 'Normal(N =' + str(normal_pt.size/2) +')')

plt.axis('tight')
plt.xlabel(feature_x, fontsize = 'large')
plt.ylabel(feature_y, fontsize = 'large')
plt.legend()
plt.suptitle(feature_x + ' vs. ' + feature_y, fontsize = 16)
plt.title('Specificity: ' + '{0:.3f}'.format(specificity) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sensitivity), fontsize = 12)

plt.show()


