
# coding: utf-8

# In[113]:

import pandas as pd
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.qda import QDA
pd.set_option('display.max_rows', 30)


# In[125]:

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


# In[115]:

normal_ndata,normal_ndim = normal_pt.shape
cancer_ndata,cancer_ndim = cancer_pt.shape


# In[116]:

#Sort given training data with corresponding labels
nor_n = np.zeros(int(normal_pt.size/normal_ndim))
can_n = np.ones(int(cancer_pt.size/cancer_ndim))
labels = np.hstack((nor_n, can_n))
train_data = np.vstack((normal_pt, cancer_pt))


# In[117]:

train_data.shape


# In[118]:

clf = QDA()
trained_clf = clf.fit(train_data, labels)
normal_pred = trained_clf.predict(normal_pt)
trueneg_n = (normal_pred == 0).sum()
specificity = trueneg_n/int(normal_ndata)


# In[119]:

cancer_pred = trained_clf.predict(cancer_pt)
truepos_n = (cancer_pred == 1).sum()
sensitivity = truepos_n/int(cancer_ndata)


# In[120]:

#Generate grids for the entire plot
xx, yy, zz = np.meshgrid(np.linspace(0, 255, 100), np.linspace(0, 255, 100), np.linspace(0, 0.2, 200))
plot_grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

#Calculate the prediction probability for each point on the grid
grid_result = clf.predict_proba(plot_grid)[:,1].reshape(xx.shape)


# In[124]:

a = abs(grid_result - 0.5)
sur_x, sur_y = np.meshgrid(np.linspace(0, 255, 100), np.linspace(0, 255, 100))
sur_z = np.zeros(sur_x.size).reshape(sur_x.shape)

sur_z.shape
for i in range(100):
    for j in range(100):     
        sur_z[i][j] = zz[i][j][a[i][j].argmin()]
sur_z


# In[126]:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1.ravel(), y1.ravel(), z1.ravel(), c = 'r', marker = 'o', label = 'Cancer')
ax.scatter(x2.ravel(), y2.ravel(), z2.ravel(), c = 'b', marker = '^', label = 'Normal')


ax.plot_wireframe(sur_x, sur_y, sur_z,  rstride=10, cstride=10, color='g')

plt.axis('tight')
plt.title('Specificity: ' + '{0:.3f}'.format(specificity) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sensitivity), fontsize = 12)
plt.suptitle('NADH intensity - FAD intensity - Heterogeneit redox', fontsize = 14)
ax.set_xlabel('nadh_intensity')
ax.set_ylabel('fad_intensity')
ax.set_zlabel('redox_hetero')
plt.legend()
plt.show()



