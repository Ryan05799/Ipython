
# coding: utf-8

# In[37]:

import os
import os.path
import pandas as pd
import numpy as np
import time
import QDAResultGen as qdar
import FeatureGen as fg


# In[38]:

#Setting parameters
#Path
listFilePath = './data/datalist.xlsx'
dataDirPath = './data/'
resultDirPath = './result/'
dataClasses = ['cancer', 'normal']
featuresInUse = ['0_data ID', 'Fad_Intensity_B', 'Fad_Intensity_G',
       'Fad_Intensity_Graylevel', 'Fad_Intensity_R', 'Fad_std_B', 'Fad_std_G',
       'Fad_std_R', 'NADH_Intensity_B', 'NADH_Intensity_G',
       'NADH_Intensity_Graylevel', 'NADH_Intensity_R', 'NADH_std_B',
       'NADH_std_G', 'NADH_std_R', 'redoxRatioGrid', 'redoxRatioPix']

useFadOnly = False
update = False



# In[39]:

#This function read the data list file
#Generate and update data feature files in each directory according to the datalist
def updateFeature(fadOnly = False):
    for sheet in range(0, 2):
        
        dataClass = dataClasses[sheet]
        
        #load data list information to dataFrame
        df = pd.read_excel(listFilePath, sheetname = sheet, header = 0)
        
        #Extract only the data marked valid (in excel) for analysis
        validData = np.array(df['Valid'])
        dataID = np.array(df['DataID'])[validData]
        location = np.array(df['Location'])[validData]

        counter = 0
        for index in range(dataID.size):
            
            dataFilePath = dataDirPath + dataClass + '/' + location[index] + '/'
            print('Processing data: ' + str(dataID[index]))
            print('Search data in ' + dataFilePath)
            currentData = fg.FeatureGen(dataID[index], dataFilePath, single = fadOnly)
            
            #Check if current data file exists, generate feature result file if it does
            if currentData.valid:
                currentData.outputResultFile()
                print('Data ' + str(dataID[index]) + ' complete')
                counter += 1
                
        print(str(counter) + ' ' +  dataClass + ' data  processed!!')


# In[40]:

#This function read the data list file, gathering the features for each data and generate data table

def dataTableGen():
    
    t = time.localtime(time.time())
    dateStr = str(t.tm_year) + str(t.tm_mon) + str(t.tm_mday) + '_' + str(t.tm_hour) + str(t.tm_min)
    #create result directory
    
    if not os.path.exists(resultDirPath + dateStr):
        os.makedirs(resultDirPath + dateStr)
        
        
    tableFileName = resultDirPath + dateStr + '/dataTable.xlsx'
    writer = pd.ExcelWriter(tableFileName)

    for sheet in range(0, 2):
        df_r = pd.read_excel(listFilePath, sheetname = sheet, header = 0)

        dataClass = dataClasses[sheet]
        #Extract only the data marked valid (in excel) for analysis
        validData = np.array(df_r['Valid'])
        dataID = np.array(df_r['DataID'])[validData]
        location = np.array(df_r['Location'])[validData]

        dfout = pd.DataFrame(columns = list(featuresInUse))

        for index in range(dataID.size):
            dataFilePath = dataDirPath + dataClass + '/' + location[index] + '/' + str(dataID[index]) + '/' + 'feature_results.xlsx'

            if os.path.isfile(dataFilePath):
                featDF = pd.read_excel(dataFilePath, sheetname = 0, header = 0)
                dfout = dfout.append(featDF, ignore_index = True)        
            else:
                print( dataClass, location[index],' data ' + str(dataID[index]) + ' Feature result not found!')

        dfout.to_excel(writer, dataClass)
    #Write to the file
    writer.save()
    return resultDirPath + dateStr + '/'
    

