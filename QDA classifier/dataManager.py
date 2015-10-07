
# coding: utf-8

# In[31]:

import os
import os.path
import pandas as pd
import numpy as np
import time
import QDAResultGen as qdar
import FeatureGen as fg
import dataTableGen as dtg


# In[32]:

df = pd.read_excel(dtg.listFilePath, sheetname = 2, header = 0)
commandTable = {'Command': 'Parameter'}

#Load commands
cmd = df['Command']
param = df['Parameter']
for i in range(cmd.size):
    key = cmd[i]
    commandTable[key] = param[i]


# In[33]:

if commandTable['update_features']:
    dtg.updateFeature(fadOnly = commandTable['Use_Fad_only'])

#Generate new data table
resultPath = dtg.dataTableGen()
print('Read table from ' + resultPath)


# In[34]:

qdaresult = qdar.QDAResultGen(resultPath)

itr = int(commandTable['CV_itr'])

print('Fad_Intensity_G vs Fad_std_G')
nor, cn = qdaresult.loadFeatureArr('Fad_Intensity_G' ,  'Fad_std_G')
QDAClf , sn, sp = qdaresult.QDATrain(nor, cn)
tag = 'Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
qdaresult.plotQda2dFig(QDAClf, nor, cn, 200, 80, 'Fad_Intensity_G', 'Fad_std_G',showfig = False, title = tag, figDir ='2DTrainOnly')


if (commandTable['CV']):
    print('Fad_Intensity_G vs Fad_std_G repeat' + str(itr) + ' times')
    qdaresult.crossValidate(itr, 'Fad_Intensity_G' , 'Fad_std_G')

if not (commandTable['Use_Fad_only']):
    
    print('Fad_Intensity_G vs redoxRatioPix')
    nor, cn = qdaresult.loadFeatureArr('Fad_Intensity_G' ,  'redoxRatioPix')
    QDAClf , sn, sp = qdaresult.QDATrain(nor, cn)
    tag = 'Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
    qdaresult.plotQda2dFig(QDAClf, nor, cn, 200, 0.15, 'Fad_Intensity_G', 'redoxRatioPix',showfig = False, title = tag, figDir ='2DTrainOnly')
    
    print('NADH_Intensity_B vs redoxRatioPix')
    nor, cn = qdaresult.loadFeatureArr('NADH_Intensity_B' ,  'redoxRatioPix')
    QDAClf , sn, sp = qdaresult.QDATrain(nor, cn)
    tag = 'Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
    qdaresult.plotQda2dFig(QDAClf, nor, cn, 200, 0.15, 'NADH_Intensity_B', 'redoxRatioPix',showfig = False, title = tag, figDir ='2DTrainOnly')

    print('Fad_Intensity_G vs redoxRatioGrid')    
    nor, cn = qdaresult.loadFeatureArr('Fad_Intensity_G' ,  'redoxRatioGrid')
    QDAClf , sn, sp = qdaresult.QDATrain(nor, cn)
    tag = 'Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
    qdaresult.plotQda2dFig(QDAClf, nor, cn, 200, 0.15, 'Fad_Intensity_G', 'redoxRatioGrid',showfig = False, title = tag, figDir ='2DTrainOnly')
    
    print('NADH_Intensity_B vs redoxRatioGrid')
    nor, cn = qdaresult.loadFeatureArr('NADH_Intensity_B' ,  'redoxRatioGrid')
    QDAClf , sn, sp = qdaresult.QDATrain(nor, cn)
    tag = 'Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
    qdaresult.plotQda2dFig(QDAClf, nor, cn, 200, 0.15, 'NADH_Intensity_B', 'redoxRatioGrid',showfig = False, title = tag, figDir ='2DTrainOnly')

    print('NADH_Intensity_B vs NADH_std_B')
    nor, cn = qdaresult.loadFeatureArr('NADH_Intensity_B' ,  'NADH_std_B')
    QDAClf , sn, sp = qdaresult.QDATrain(nor, cn)
    tag = 'Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
    qdaresult.plotQda2dFig(QDAClf, nor, cn, 200, 80, 'NADH_Intensity_B', 'NADH_std_B',showfig = False, title = tag, figDir ='2DTrainOnly')

    if(commandTable['Show_3D']):
        print('Fad_Intensity_G vs redoxRatioPix vs redoxRatioPix')
        nor3d , cn3d = qdaresult.loadFeatureArr('Fad_Intensity_G' , 'NADH_Intensity_B',  'redoxRatioPix')
        QDAClf , sn, sp = qdaresult.QDATrain(nor3d, cn3d)
        tag = 'Inside test Specificity: ' + '{0:.3f}'.format(sp) +  ' ; ' + 'Sensitivity:' + '{0:.3f}'.format(sn)
        qdaresult.plotQda3dFig(QDAClf, nor3d , cn3d, title = tag)
        qdaresult.crossValidate(itr, 'Fad_Intensity_G', 'NADH_Intensity_B' , 'redoxRatioPix')
        
    if(commandTable['CV']):
        print('Fad_Intensity_G vs redoxRatio repeat' + str(itr) + ' times')
        qdaresult.crossValidate(itr, 'Fad_Intensity_G' , commandTable['RedoxRatio'])
        
        print('NADH_Intensity_B vs NADH_std_B repeat' + str(itr) + ' times')
        qdaresult.crossValidate(itr, 'NADH_Intensity_B' , 'NADH_std_B')
        
        print('NADH_Intensity_B vs redoxRatio repeat' + str(itr) + ' times')
        qdaresult.crossValidate(itr, 'NADH_Intensity_B' , commandTable['RedoxRatio'])

    if(commandTable['Five_dimQDA']):
        
        print('5D QDA')
        qdaresult.crossValidate(itr, 'Fad_Intensity_G', 'Fad_std_G', 'NADH_Intensity_B' , 'NADH_std_B', commandTable['RedoxRatio'])

    if(commandTable['Weak_learner']):
        print('Weak Learner')
        perc = commandTable['WL_percentage']
        perc_trn = commandTable['WL_percentage_tr']
        print('3D')
        qdaresult.crossValidateWL( itr, 'Fad_Intensity_G', 'NADH_Intensity_B' , commandTable['RedoxRatio'], perc = perc, perc_trn = perc_trn)
        print('5D')
        qdaresult.crossValidateWL( itr, 'Fad_Intensity_G', 'Fad_std_G', 'NADH_Intensity_B' , 'NADH_std_B', commandTable['RedoxRatio'], 
                                  perc = perc, perc_trn = perc_trn)
        
print('Done')
os.system("pause")

