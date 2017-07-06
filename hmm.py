
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
from __future__  import print_function
import pandas as pd
from HiddenMarkovModel import HiddenMarkovModel
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
get_ipython().magic(u'matplotlib inline')


# In[5]:

# 定义预处理函数，读入数据预处理，包括取相关列，计算RSSI以及去除nan值
def pre_processing(filename):
    file_2g = pd.read_csv(filename).loc[:,['MRTime','Grid_center_x','Grid_center_y','RNCID_1', 'CellID_1','EcNo_1', 'RSCP_1', 'RNCID_2', 'CellID_2','EcNo_2', 'RSCP_2','RNCID_3', 'CellID_3','EcNo_3', 'RSCP_3', 'RNCID_4', 'CellID_4','EcNo_4', 'RSCP_4', 'RNCID_5', 'CellID_5','EcNo_5', 'RSCP_5', 'RNCID_6', 'CellID_6','EcNo_6', 'RSCP_6','Grid_ID']]
    gong_can = pd.read_csv("final_2g_gongcan.csv", encoding='gbk').loc[:,['CGI', 'LAC', 'CI', u'经度', u'纬度']]
    
    for i in range(6):
        RNCID = "RNCID_" + str(i+1)
        CellID = "CellID_" + str(i+1)
        station = pd.merge(file_2g, gong_can, left_on=[RNCID, CellID], right_on=['LAC','CI'], how='inner')[['CGI']]
        file_2g[[str(i+1)+'station']] = station
        file_2g['RSSI'+str(i+1)] = file_2g['EcNo_' + str(i+1)] - file_2g['RSCP_' + str(i+1)]

    merge_data = file_2g
    
    merge_data['MRTime'] = merge_data['MRTime'].str.split()
    timeCol = pd.Series(map(lambda x:x[1],merge_data['MRTime']),index=merge_data.index)
    timeCol = pd.to_datetime(timeCol,format="%H:%M:%S.%f")
    merge_data['MRTime'] = timeCol
    
    merge_data = merge_data.sort_values(by=['MRTime'])
    
    for i in range(1,7):
        del merge_data['EcNo_' + str(i)],merge_data['RSCP_' + str(i)]
        
    for i in range(1,7):
        del merge_data['RNCID_' + str(i)],merge_data['CellID_' + str(i)]
        
    return merge_data


# In[6]:

#读取数据
train_data = pd.read_csv('train_data_after_process.csv',index_col="Unnamed: 0").drop(78066)
test_data = pre_processing('final_2g_te.csv')


# In[7]:

#去某条轨迹，如果这里去了第三条轨迹做示例
trace = pd.read_csv('newtrace.csv',index_col="Unnamed: 0")
del trace["Unnamed: 0.1"]
trace['MRTime'] = pd.to_datetime(trace['MRTime'],format="%Y-%m-%d %H:%M:%S.%f")
traj_1 = trace[trace['path_index'] == 3]


# In[8]:

station_compared_traj = []
rssi_code_traj = []
for i in range(1, 6 + 1):
    station_compared_traj.append('Station_' + str(i))
    rssi_code_traj.append('RSSI_' + str(i))


# In[9]:

# 计算emission score的中间步骤
observables = []
new_pattern = train_data[station_compared_traj]
big_mat = train_data[rssi_code_traj]
for row in traj_1.index:
    #取出test数据集中的某一条记录
    fnew = traj_1.loc[row]
    #取出该记录能在共参表里匹配到的基站
    ls = fnew[station_compared_traj].dropna().tolist()
    #取出改记录的RSSI向量，如果没有信号则置为零
    rssi = np.array(fnew[rssi_code_traj].tolist())
    #取出匹配的中间0/1矩阵，做加速用，算是小trick
    after_process = new_pattern[new_pattern.isin(ls)].dropna(axis=0,how='all').fillna(0) != 0
    #得到train数据中匹配的RSSI矩阵
    rssi_matrix = big_mat.loc[after_process.index]
    #算出匹配后的rssi矩阵
    fnewMatrix = after_process*rssi
    patternMatrix = after_process*rssi_matrix.values
    
    #定义一些中间量
    traj_sum = 'traj_sum' + str(row)
    M = 'M' + str(row)
    observable = 'obeservable_' + str(row)
    #定义观察链
    observables.append(observable)
    #计算欧式距离
    train_data[traj_sum] = np.sqrt(((fnewMatrix - patternMatrix)**2).sum(axis=1))
    #动态得到dRmax
    max_rssi = train_data[traj_sum].max()
    #得到匹配基站数M
    train_data[M] = after_process.sum(axis=1)
    #初始化
    train_data[observable] = 0
    middleware = train_data[train_data[M] > 0]
    #使用M×3+（dRmax - dR(F1,F2)）
    train_data.loc[middleware.index,observable] = middleware[M] * 3 + (max_rssi - middleware[traj_sum] / middleware[M])


# In[12]:

# emision矩阵
groups = train_data.groupby(['Grid_ID'])
emission = groups[observables].agg(np.max).T
emission_matrix = ((emission - emission.min()) / (emission.max() - emission.min()))
                    .dropna(axis = 1) #min-max normalizd


# In[14]:

#匹配基站坐标,对后面做匹配用
Grid_ID = train_data[['Grid_ID','Grid_center_y','Grid_center_x']].drop_duplicates().dropna()
Grid_ID.index = Grid_ID['Grid_ID']
del Grid_ID['Grid_ID']


# In[17]:

#trans矩阵
manhattan = Grid_ID.loc[emission_matrix.columns].values
trans = manhattan_distances(manhattan,manhattan)

#曼哈顿距离计算，haversine作为垂直和水平距离
manhattan_ls = []
for i in manhattan:
    tmp_ls = []
    for j in manhattan:
        vertical = haversine(i[1],i[0],i[1],j[0])
        horizontal = haversine(i[1],i[0],j[1],i[0])
        tmp_ls.append((np.abs(vertical) + np.abs(horizontal))/30.0)
    manhattan_ls.append(tmp_ls)
trans = np.array(manhattan_ls)

#将对角线置为1
np.fill_diagonal(trans,1)
trans_matrix = pd.DataFrame(trans,index=emission_matrix.columns,columns=emission_matrix.columns)
trans_matrix_reverse = 1/trans_matrix


# In[19]:

#做成输入的标准矩阵
emi_mat = emission_matrix.as_matrix()
trans_mat = trans_matrix_reverse.as_matrix()


# In[20]:

#初始概率，只展示效果较好的初始概率，即平均
allNumber = len(trans_mat)
p0 = [1.0/allNumber for i in range(allNumber)]


# In[22]:

#定义模型
model =  HiddenMarkovModel(trans_mat, emi_mat, p0)


# In[23]:

#模型训练
states_seq, state_prob = model.run_viterbi([i for i in range(len(emission_matrix))],summary=True)


# In[56]:

#回溯匹配
grid = emission_matrix.columns[states_seq]
predict = Grid_ID.loc[grid]#.values


# In[52]:

original_ls = traj_1[['Grid_center_y','Grid_center_x']].values.tolist()
predict_ls = predict.tolist()


# In[40]:

#误差计算，使用haversine距离
error_ls = []
for (x,y) in zip(original,predict):
    error_ls.append(haversine(x[1],x[0],y[1],y[0]))
    
#平均误差
error_mean = np.mean(ls)
#做图
plt.plot(error_ls)


# In[ ]:

predict.loc['Grid_ID'] = grid
predict.to_csv("hmm2Input.csv")


# In[9]:

from math import radians, cos, sin, asin, sqrt  
      
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）  
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  
    # 将十进制度数转化为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
      
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r * 1000  


# In[10]:

# HiddenMarkov.py Simple Example

p0 = np.array([0.6,0.4])
emi = np.array([[0.5,0.1],
               [0.4,0.3],
               [0.1,0.6]])
trans = np.array([[0.7,0.3],
                 [0.4,0.6]])
states = {0:'Healthy',1:'Fever'}

obs = {0:'normal',1:'cold',2:'dizzy'}

obs_seq = np.array([0,0,1,2,2])

df_p0 = pd.DataFrame(p0,index=['Healthy','Fever'],columns=['Prob'])
df_emi = pd.DataFrame(emi,index=['Normal','Cold','Dizzy'],columns=['Healthy','Fever'])
df_trans = pd.DataFrame(trans,index=['fromHealthy','fromFever'],columns=['toHealthy','toFever'])



model =  HiddenMarkovModel(trans, emi, p0)
states_seq, state_prob = model.run_viterbi(obs_seq, summary=True)

print(states_seq)


# In[12]:

# author : 夏陈 ，洪嘉勇
# 如有疑问，欢迎发邮件 stanforxc@gmail.com

