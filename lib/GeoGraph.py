from torch_geometric.data import Data
import pandas as pd
import numpy as np
import math
import multiprocessing
import torch
from filelock import FileLock
import os
Base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
def calculate_distance(lat1, lon1, lat2, lon2):
    # 使用大圆公式计算两点之间的距离
    radius = 6378.137   # 地球半径，单位：千米

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance
def geohash_decode(code):
        # Base32解码
        code_bin = ''
        for s in code:
            num = Base32.find(s)
            code_bin += bin(num)[2:].rjust(5, '0')
        # 分离经纬度
        lng_str, lat_str = '', ''
        for i in range(len(code_bin)):
            if i % 2 == 0:
                lng_str += code_bin[i]
            else:
                lat_str += code_bin[i]

        # 二进制解码
        longitudes, latitudes = [[-180, 180]], [[-90, 90]]
        for s in lng_str:
            left, right = longitudes[-1]
            if s == '0':
                longitudes.append([left, (left + right) / 2])
            else:
                longitudes.append([(left + right) / 2, right])
        for s in lat_str:
            left, right = latitudes[-1]
            if s == '0':
                latitudes.append([left, (left + right) / 2])
            else:
                latitudes.append([(left + right) / 2, right])

        lng = (longitudes[-1][0] + longitudes[-1][0]) / 2
        lat = (latitudes[-1][0] + latitudes[-1][0]) / 2
        return lat, lng
def geohash_encode(lat, lng, n):
    # 二进制编码
    lat_num = (5 * n) // 2
    lng_num = 5 * n - lat_num
    lng_str, lat_str = '', ''
    longitudes, latitudes = [[-180, 180]], [[-90, 90]]
    for _ in range(lng_num):
        left, right = longitudes[-1]
        if lng < (left + right) / 2:
            longitudes.append([left, (left + right) / 2])
            lng_str += '0'
        else:
            longitudes.append([(left + right) / 2, right])
            lng_str += '1'
    for _ in range(lat_num):
        left, right = latitudes[-1]
        if lat < (left + right) / 2:
            latitudes.append([left, (left + right) / 2])
            lat_str += '0'
        else:
            latitudes.append([(left + right) / 2, right])
            lat_str += '1'

    # 交叉合并
    str_bin = ''
    for i in range(5 * n):
        if i % 2 == 0:
            str_bin += lng_str[i // 2]
        else:
            str_bin += lat_str[i // 2]

    # Base32编码
    code = ''
    for i in range(n):
        code += Base32[int(str_bin[i * 5: i * 5 + 5], 2)]

    return code
def encode_scale(lat,lng,lat_mean,lng_mean,lat_diff_mean,lng_diff_mean):#由于经纬度精度要求较高所以要放大差距
    p=1
    scale = 1
    min_lat=lat_mean-p*lat_diff_mean
    max_lat=lat_mean+p*lat_diff_mean

    new_lat=((lat-min_lat)*scale)/(max_lat-min_lat)

    min_lng=lng_mean-p*lng_diff_mean
    max_lng=lng_mean+p*lng_diff_mean
    new_lng = ((lng- min_lng)*scale) / (max_lng- min_lng)
    return new_lat,new_lng
def decode_scale(lat,lng,lat_mean,lng_mean,lat_diff_mean,lng_diff_mean):
    p=1
    scale=1
    min_lat = lat_mean - p * lat_diff_mean
    max_lat = lat_mean + p * lat_diff_mean
    new_lat=((lat* (max_lat - min_lat))/scale)+min_lat
    min_lng = lng_mean - p * lng_diff_mean
    max_lng = lng_mean + p * lng_diff_mean
    new_lng= ((lng * (max_lng - min_lng)) / scale) + min_lng
    return new_lat, new_lng
def get_feature(geo,geodic, c_lat,c_lng,dataset, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean):#获取节点属性
    # geohsh的中心
    #geo_lat1, geo_lng1 = encode_scale(geo_lat, geo_lng, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
    if len(dataset)==0:#如果这个节点没有点直接0填充
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    geo_lat  = geodic[geo][0]#geohash6lat
    geo_lng   = geodic[geo][1]
    dataset=np.array(dataset)
    #geohash6内所有点的中心
    center_latitude = np.mean(dataset[:, 0])#geo点中心
    center_longitude = np.mean(dataset[:, 1])#geo点中心
    center_latitude1, center_longitude1 = encode_scale(center_latitude, center_longitude, lat_mean, lng_mean,lat_diff_mean, lng_diff_mean)
    dis=calculate_distance(geo_lat,geo_lng, center_latitude, center_longitude)#据geohash6的距离


    #距离整个图中心的距离，整体
    distances_center = [calculate_distance(point[0],point[1],c_lat,c_lng) for point in dataset]#距离整个点中心的距离
    coverage_radius_center = np.mean(distances_center)#散点分布
    max_center_distance_center=np.max(distances_center)
    min_center_distance_center = np.min(distances_center)
    std_center_distance_center=np.std(distances_center)#分散程度

    if len(dataset)==1:
        #return str(center_latitude1)+"@"+str(center_longitude1)+"@"+str(dis)+"@"+str(coverage_radius_center)+"@"+str(max_center_distance_center)+"@"+str(min_center_distance_center)+"@"+str(std_center_distance_center) #[center_latitude1,center_longitude1,dis,coverage_radius_center,max_center_distance_center,min_center_distance_center,std_center_distance_center,0,0,0,0,1,0,0,0,0,0]长度7
        return [center_latitude1,center_longitude1,dis,coverage_radius_center,max_center_distance_center,min_center_distance_center,std_center_distance_center,0,0,0,0,1,0,0,0,0,0]

    #geohash内部点到geohash内部点中心的距离，代表点在geohash内部的分散程度
    distances = [calculate_distance(point[0],point[1],center_latitude, center_longitude) for point in dataset]
    coverage_radius = np.mean(distances)#散点分布
    max_center_distance=np.max(distances)
    min_center_distance = np.min(distances)
    std_center_distance=np.std(distances)
    pointsnum=len(dataset)
    if coverage_radius==0:
        density=0
    else:
        density=pointsnum/(coverage_radius*coverage_radius*3.14)#每个点的面积

    #geohash内部 点与点之间的距离，代表点紧密程度
    dist_matrix2 = []
    for i in range(len(dataset)):
        for j in range(i+1,len(dataset)):
            dist_matrix2.append(calculate_distance(dataset[i][0],dataset[i][1],dataset[j][0],dataset[j][1]))
    average_distance = np.mean(dist_matrix2)
    std_distance = np.std(dist_matrix2)
    max_distance = np.max(dist_matrix2)
    min_distance = np.min(dist_matrix2)
    return [center_latitude1,center_longitude1,dis,coverage_radius_center,max_center_distance_center,min_center_distance_center,std_center_distance_center,coverage_radius,max_center_distance,min_center_distance,std_center_distance,pointsnum,density,average_distance,std_distance,max_distance,min_distance]

def combinations(n,limit):
    vail_combine=[]
    for i in range(n):
        combine=[]
        if i+limit<n:
            for j in range(i,i+limit):
                combine.append(j)
            vail_combine.append(combine)
        else:
            break
    return vail_combine
def get_node(geolist,geo_group,geodic, c_lat,c_lng,lat_mean, lng_mean, lat_diff_mean, lng_diff_mean):#获取图节点
    node=[]#根据geohash得到这天的图节点
    geo_point_list=[]#这天的geohash对应的点
    for i in range(len(geolist)):
        #geolist[i]代表此次的geohash，geodic代表所有geohash对应的经纬度减少计算，geo_group[geolist[i]]代表geohash对应的点
        node.append(get_feature(geolist[i],geodic,c_lat,c_lng,geo_group[geolist[i]],lat_mean, lng_mean, lat_diff_mean, lng_diff_mean))
        geo_point_list.append(geo_group[geolist[i]])#把分组点加入列表
    return  node,geo_point_list
def get_edge(geo_group,disf):#获取图的边
    geolist=list(geo_group.keys())
    edge_index = []
    edge_attr = []
    for i in range(len(geolist)):
        if len(geo_group[geolist[i]])==0:#如果geohash没有点直接没有边
           continue
        else:
            for j in range(i+1,len(geolist)):#如果geohash没有点直接没有边
              if  len(geo_group[geolist[j]])==0:
                  continue
              else:
                  # 第1个geohash的中心经纬度
                  p1_lat=np.mean(np.array(list(geo_group[geolist[i]]))[:, 0])
                  p1_lng = np.mean(np.array(list(geo_group[geolist[i]]))[:, 1])
                  # 第2个geohash的中心经纬度
                  p2_lat=np.mean(np.array(list(geo_group[geolist[j]]))[:, 0])
                  p2_lng = np.mean(np.array(list(geo_group[geolist[j]]))[:, 1])
                  #计算距离，如果距离小于2km那么有边关系，添加边，以及边属性为距离
                  dis=calculate_distance(p1_lat, p1_lng, p2_lat, p2_lng)
                  if dis<disf:#原始小于2
                      edge_index.append([i,j])
                      edge_attr.append(dis)
                      edge_index.append([j,i])
                      edge_attr.append(dis)
    return edge_index,edge_attr
def get_bound(bound):
    boundlist=bound.split("_")
    return boundlist
def get_aug(aug):
        auglist=[]
        aug_data=aug.split("_")
        for d2 in aug_data:
            d2=d2.split("@")
            auglist.append(d2)
        return auglist
def get_graph(geolist,geodic,aug,lat_mean,lng_mean,lat_diff_mean,lng_diff_mean):
        # 遍历这一天所有点，分类这些点到属于自己的geohash6的分组，构建ggeohash6的分组
        disf=2
        lens=6
        geo_group = {}
        lats = []
        lngs = []
        for geo in geolist:#初始化geohash分组
            geo_group[geo]=[]
        for lat,lng in aug:#遍历所有点
            lat=float(lat)
            lng=float(lng)
            lats.append(lat)
            lngs.append(lng)
            geo=geohash_encode(lat,lng,lens)
            geo_group[geo].append([lat,lng])
        #获取这一天的中心
        c_lat=np.mean(lats)
        c_lng=np.mean(lngs)
        #通过geogroup获取节点，边以及边属性
        node,geo_point_list = get_node(geolist,geo_group, geodic, c_lat,c_lng,lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)  # 获取节点
        c_lat,c_lng=encode_scale(c_lat, c_lng, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
        edge_index,edge_attr=get_edge(geo_group,disf)#获取边,disf为获取边的节点之间的距离
        return [c_lat,c_lng,node,edge_index,edge_attr,geolist,geo_point_list]
def get_Graph(path_aug,path_bound,outpath,ip,date_interval):#获取训练数据
            #获取图的边界信息
            df_bound = pd.read_csv(path_bound + "/" + ip + ".csv", header=0)
            bound = get_bound(df_bound.iloc[0]["boundary"])
            lat_mean = df_bound.iloc[0]["lat_mean"]
            lng_mean = df_bound.iloc[0]["lng_mean"]
            lat_diff_mean = df_bound.iloc[0]["lat_diff_mean"]
            lng_diff_mean = df_bound.iloc[0]["lng_diff_mean"]
            geodic = {}  # 初始化geohash字典对应的经纬度
            for geo in bound:
                lat, lng = geohash_decode(geo)
                geodic[geo] = [lat, lng]
            geolist = list(geodic.keys())#获取所有的geohash列表，并确定顺序
            #获取点的数据增强数据
            df_aug=pd.read_csv(path_aug + "/" + ip + ".csv",index_col=False)
            dates = df_aug.columns.tolist()
            dates = dates[1:]
            date_length=len(dates)
            combination = combinations(date_length, date_interval)  # 用7天的数据去预测第8天的日期,29天内这样的组合
            train_list = []
            label_list = []
            padding_mask_list = []
            current_label_list=[]
            geo_point_list_train_list=[]
            for index,row in df_aug.iterrows():#每个IP数据增强30天的数据 一共有100条，遍历这100条l
                graph_dic = {}
                for i in range(date_length):
                    aug=get_aug(row[str(i)])#获取数据增强数据从地0天到第29天
                    graph_dic[i]=get_graph(geolist,geodic,aug,lat_mean,lng_mean,lat_diff_mean,lng_diff_mean)#获取29天的图数据
                #根据combination 从30天选取连续7天的组合
                for comb in combination:
                    train = []
                    label = []
                    padding_mask = []
                    current_label=[]
                    geo_point_list_train=[]
                    for index in comb:#某一天的图
                        latc = graph_dic[index][0]
                        lngc = graph_dic[index][1]
                        x = graph_dic[index][2]
                        edge_index=graph_dic[index][3]
                        edge_attr=graph_dic[index][4]
                        geo_point_list=graph_dic[index][6]
                        x = torch.tensor(x , dtype=torch.float32)  # 使用经纬度作为节点特征
                        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
                        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                        train.append(data)
                        current_label.append([latc, lngc])
                        geo_point_list_train.append(geo_point_list)
                    for index in range(comb[-1], comb[-1] + 8):  #
                        if index <= date_length-1:
                            latc = graph_dic[index][0]
                            lngc = graph_dic[index][1]
                            label.append([latc, lngc])
                            padding_mask.append(0)
                        else:
                            label.append([0, 0])
                            padding_mask.append(1)
                    geo_point_list_train_list.append(geo_point_list_train)
                    train_list.append(train)#把7天的数据加入一条训练
                    label_list.append(torch.tensor(np.array(label), dtype=torch.float32))
                    padding_mask_list.append(torch.tensor(np.array(padding_mask), dtype=torch.bool))
                    current_label_list.append(torch.tensor(np.array(current_label), dtype=torch.float32))
            lock = FileLock(outpath + "/" + ip + ".pt")
            with lock:
                torch.save((train_list, label_list, padding_mask_list,current_label_list,geolist,geo_point_list_train_list), outpath+"/"+ ip + ".pt")
                print(f"Graph_Processed {ip}")
                return 0
