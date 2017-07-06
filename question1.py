
# coding: utf-8

# In[1]:


import pandas as pd
from numpy import nan
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import euclidean_distances


# In[2]:

#读取数据
final_2g_tr = pd.read_csv('final_2g_tr.csv',encoding="gbk")
final_2g_tr = final_2g_tr.dropna()
final_2g_gongcan = pd.read_csv('final_2g_gongcan.csv',encoding="gbk")


# In[3]:

#读取数据
final_2g_te = pd.read_csv('final_2g_te.csv',encoding="gbk")
final_2g_te = final_2g_te.dropna()


# In[4]:

for i in range(1,7):
    final_2g_tr['STRENGTH_'+str(i)] = final_2g_tr['RSCP_'+str(i)] - final_2g_tr['EcNo_'+str(i)]
    final_2g_te['STRENGTH_'+str(i)] = final_2g_te['RSCP_'+str(i)] - final_2g_te['EcNo_'+str(i)]


# In[5]:

final_2g_tr['relativeLatitude'] = -1
final_2g_tr['relativeLongitude'] = -1
final_2g_te['relativeLatitude'] = -1
final_2g_te['relativeLongitude'] = -1


# In[6]:

group_final_2g_te = final_2g_tr.groupby(['SRNCID','BestCellID'])
groups_final_2g_te = group_final_2g_te.groups

group_final_2g_gongcan = final_2g_gongcan.groupby(['LAC','CI'])
groups_final_2g_gongcan = group_final_2g_gongcan.groups

group_final_2g_tr = final_2g_tr.groupby(['SRNCID','BestCellID'])
groups_final_2g_tr = group_final_2g_tr.groups


# In[8]:

from sklearn.ensemble import RandomForestRegressor


# In[ ]:

# training
# 第一问第一题，做出所有随机森林estimation模型
estimators = {}
for key,group in group_final_2g_tr:
    index = groups_final_2g_gongcan[key][0]
    tmpLatitude = final_2g_gongcan[u"纬度"][index]
    tmpLongitude = final_2g_gongcan[u"经度"][index]
    group.dropna()
    group['relativeLongitude'] = group['Longitude'] - tmpLongitude
    group['relativeLatitude'] = group['Latitude'] - tmpLatitude
    y_train = group[['relativeLatitude','relativeLongitude']]
    X_train = group[['BestCellID', 'SRNCID', 'RNCID_1', 'RNCID_2', 'RNCID_3', 'RNCID_4', 'RNCID_5', 'RNCID_6', 'RSCP_1', 'RSCP_2',
           'RSCP_3', 'RSCP_4', 'RSCP_5', 'RSCP_6', 'EcNo_1', 'EcNo_2', 'EcNo_3', 'EcNo_4', 'EcNo_5', 'EcNo_6',
           'STRENGTH_1', 'STRENGTH_2', 'STRENGTH_3', 'STRENGTH_4', 'STRENGTH_5', 'STRENGTH_6']]
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    estimator.fit(X_train, y_train)
    estimators[key] = estimator


# In[ ]:

# test
# 第一问第二题，计算回归定位算法
added_group = {}
for key,group in group_final_2g_tr:
    index = groups_final_2g_gongcan[key][0]
    tmpLatitude = final_2g_gongcan[u"纬度"][index]
    tmpLongitude = final_2g_gongcan[u"经度"][index]
    group.dropna()
    group['relativeLongitude'] = group['Longitude'] - tmpLongitude
    group['relativeLatitude'] = group['Latitude'] - tmpLatitude
    estimator = estimators[key]
    y_test = group[['relativeLatitude','relativeLongitude']]
    X_test = group[['BestCellID', 'SRNCID', 'RNCID_1', 'RNCID_2', 'RNCID_3', 'RNCID_4', 'RNCID_5', 'RNCID_6', 'RSCP_1', 'RSCP_2',
           'RSCP_3', 'RSCP_4', 'RSCP_5', 'RSCP_6', 'EcNo_1', 'EcNo_2', 'EcNo_3', 'EcNo_4', 'EcNo_5', 'EcNo_6',
           'STRENGTH_1', 'STRENGTH_2', 'STRENGTH_3', 'STRENGTH_4', 'STRENGTH_5', 'STRENGTH_6']]
    
    y_predict = estimator.predict(X_test)
    s = pd.DataFrame(y_predict, index=group.index,columns=['restoreLatitude','restoreLongitude'])
    s['restoreLatitude'] += tmpLatitude
    s['restoreLongitude'] += tmpLongitude
    group = pd.concat([group, s], axis=1)
    group['deltaDist'] = -1
    group['deltaDist'] = euclidean_distances(group[['Latitude','Longitude']],group[['restoreLatitude','restoreLongitude']])
    added_group[key] = group
    


# In[11]:

#第一问第三题，模型迁移，交叉验证
cross_group_estimation = {}

for K in added_group:
    group = added_group[K]
    y_test = group[['relativeLatitude','relativeLongitude']]
    X_test = group[['BestCellID', 'SRNCID', 'RNCID_1', 'RNCID_2', 'RNCID_3', 'RNCID_4', 'RNCID_5', 'RNCID_6', 'RSCP_1', 'RSCP_2',
                    'RSCP_3', 'RSCP_4', 'RSCP_5', 'RSCP_6', 'EcNo_1', 'EcNo_2', 'EcNo_3', 'EcNo_4', 'EcNo_5', 'EcNo_6',
                    'STRENGTH_1', 'STRENGTH_2', 'STRENGTH_3', 'STRENGTH_4', 'STRENGTH_5', 'STRENGTH_6']]
    values = {}
    for key in estimators:
        if K != key:
            esti = estimators[key]
    
            y_predict = esti.predict(X_test)
        
            estimator_label = str(key)
            
            s = pd.DataFrame(y_predict, index=group.index,columns=['restoreLatitude' + estimator_label,'restoreLongitude' + estimator_label])
            s['restoreLatitude' + estimator_label] += tmpLatitude
            s['restoreLongitude' + estimator_label] += tmpLongitude
            group = pd.concat([group, s], axis=1)
            
            group['deltaDist' + estimator_label] = -1
            group['deltaDist' + estimator_label] = euclidean_distances(group[['Latitude','Longitude']],group[['restoreLatitude' + estimator_label,'restoreLongitude' + estimator_label]])
            
    values[key] = group
    cross_group_estimation[K] = values           


# In[2]:

# author : 夏陈， 洪嘉勇
# 有问题欢迎提问， 邮箱: stanforxc@gmail.com

