
# coding: utf-8

# In[1]:


from osmread import parse_file, Node,Way,Relation
import json
from pandas.io.json import json_normalize
import pandas as pd
import math
import random
from sklearn.metrics import euclidean_distances
import numpy as np
from HiddenMarkovModel import HiddenMarkovModel
import random


# In[2]:

#序列化
node_ls = []
way_ls = []
relation = []
for entity in parse_file('map.osm'):
    if isinstance(entity, Node):
        node_ls.append(json.loads(json.dumps(entity.__dict__)))
    if isinstance(entity,Way):
        way_ls.append(json.loads(json.dumps(entity.__dict__)))
    if isinstance(entity,Relation):
        relation.append(entity.__dict__)


# In[3]:

#取出node特征
node = json_normalize(node_ls)[['id','lat','lon','timestamp','uid']]
node.index = node['id']
del node['id']


# In[4]:

#取出way特征
way = json_normalize(way_ls)[['id','nodes','timestamp','uid']]
way.index = way['id']
del way['id']


# In[5]:

#计算road segment 矩阵
def location(row):
    ls = row['nodes']
    return [node.loc[ls,['lat','lon']].values]
way['traj_long_la'] = way.apply(location,axis = 1)


# In[7]:

#读入某条轨迹数据
traj_0 = pd.read_csv('chazhiyihou.csv',index_col='Unnamed: 0')


# In[8]:

#轨迹 HM 和 HT
HM = [random.randint(0, 1) for i in range(0,len(traj_0))]
HT = [random.randint(0, 1) for i in range(0,len(traj_0))]
traj_0['HM'] = HM
traj_0['HT'] = HT


# In[9]:

#道路 HM 和 HT
wayHM = [random.randint(0, 1) for i in range(0,len(way))]
wayHT = [random.randint(0, 1) for i in range(0,len(way))]
way['HM'] = wayHM
way['HT'] = wayHT


# In[10]:

#简单拆分
traj_0_new = traj_0[['Grid_center_y','Grid_center_x','HM','HT']].dropna()
traj_0_location = traj_0_new[['Grid_center_y','Grid_center_x']]
traj_0_hh = traj_0_new[['HM','HT']]


# In[14]:

#计算每个点到road segment的距离
#尝试两种距离计算方法，一种是VTrack论文中的，另一种是CTrack论文中
def closest_distance(row):
    traj_long_la = row['traj_long_la'][0]
    traj_mat = traj_0_location.values
    ls = []
    for i in traj_mat:
        Max = -1
        for j in traj_long_la:
            Max = max(haversine(i[1],i[0],j[1],j[0]),Max)
        ls.append(Max)
    variance = np.std(np.array(ls))
    gaussian_dist = np.random.normal(0, variance, len(ls))
    return pd.Series(gaussian_dist)

def closest_distance_nicai_version(row):
    traj_long_la = row['traj_long_la'][0]
    traj_mat = traj_0_location.values
    ls = []
    for i in traj_mat:
        Max = -1
        for j in traj_long_la:
            Max = max(haversine(i[1],i[0],j[1],j[0]),Max)
        ls.append(Max)
    dist = 1/np.array(ls)
    gaussian_dist = np.power(np.e,-dist**2)
    return pd.Series(gaussian_dist)
#得到emission矩阵
dis_min_df = way.apply(closest_distance,axis=1)
emission_matrix = dis_min_df.T
emission_matrix = ((emission_matrix - emission_matrix.min()) / (emission_matrix.max() - emission_matrix.min())).dropna(axis = 1)


# In[16]:

way_mat = way["nodes"]


# In[27]:

#计算路网，当首尾相连或中间相连，即是路段相连
dic = {}
dic_concat = {}
for i in way_mat.index:
    tmp_ls = []
    concat = []
    arr = way_mat.loc[i]
    end_point = arr[-1]
    for j in way_mat.index:
        arr1 = way_mat.loc[j]
        if end_point in arr1:
            concat.append(j)
            if arr1.index(end_point) == 0:
                tmp_ls.append(j)
    dic[i] = tmp_ls
    dic_concat[i] = concat


# In[42]:

#计算状态转移矩阵
trans = pd.DataFrame(0,index = way.index,columns= way.index)
for key in dic_concat:
    #e = 1.0/(len(dic_concat[key])+1)
    trans.loc[key, dic_concat[key]] = 1
    #trans.loc[key, dic_concat[key]] = e
   #trans.loc[dic[key],key] = e
    trans.loc[key, key] = 1


# In[44]:

# 取得输入emission矩阵和transition矩阵
emi_mat = emission_matrix.values
tran_mat = trans.values
#取得初始概率
allNumber = len(tran_mat)
p0 = [1.0/allNumber for i in range(allNumber)]


# In[45]:

model = HiddenMarkovModel(tran_mat,emi_mat,p0)


# In[46]:

states_seq, state_probs = model.run_viterbi([i for i in range(len(emi_mat))],summary=True)


# In[47]:

states_seq


# In[ ]:

main = states_seq[0]
ls = [main]
for i in range(len(states_seq)):
    if main == states_seq[i]:
        continue
    else:
        ls.append(states_seq[i])
        main = states_seq[i]


# In[49]:

wayid = emission_matrix.columns[ls]
road_segment = way.loc[wayid]


# In[50]:

haha = []
for i in road_segment['traj_long_la']:
    haha.append(i[0].tolist())


# In[13]:

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

