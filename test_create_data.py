import os
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import math
import torch
import random
from additional_compare_experience.TSG.lib.LNGeoAugment import data_augmentation
from additional_compare_experience.TSG.lib.GeoGraph import get_bound,get_graph
from datetime import datetime
datelist = ["20230420","20230421","20230422","20230423","20230424","20230425","20230426","20230427","20230428","20230429","20230430","20230501",
                     "20230502","20230503","20230504","20230505","20230506","20230507","20230508","20230509","20230510","20230511",
                     "20230512","20230513","20230514","20230515","20230516","20230517","20230518","20230519"]
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
test_datelist=["20230520","20230521","20230522","20230523","20230524","20230525","20230526","20230527"]

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

def process_test(s):
    s=s.split("_")
    dics={}
    for item in s:
        l=item.split(":")
        date=l[0]
        date = test_datelist.index(date)
        lat,lng=geohash_decode(l[1])
        dics[str(date)]=[lat,lng]
    return dics

def process_geolist(geostr):
    geos = geostr.split("@")
    geol = []
    lats = []
    lngs = []
    for geo in geos:
        lat, lng = geohash_decode(geo)
        lats.append(lat)
        lngs.append(lng)
        geol.append([lat, lng])
    data = geol
    data_center = [np.mean(lats), np.mean(lngs)]
    return  data,data_center
def get_test(path,path_bound,outpath,region):
    df = pd.read_csv(path)
    df_aug={}
    df_test={}
    df_center={}
    for index, row in df.iterrows():
                ip = row["IP"]
                geolist = row["geolist"]
                geolist = geolist.split("_")
                dics={}
                dics2={}
                for geo in geolist:  # 遍历每一天
                    geo = geo.split(":")
                    date = geo[0] # 每一天
                    date = datelist.index(date)
                    data = geo[1]  # 数据
                    data, data_center = process_geolist(data)  # 获取n天的点和中心
                    max_distance = random.uniform(0.05, 1)
                    points_data = data_augmentation(data, max_distance)
                    dics[str(date)]=points_data
                    dics2[str(date)] = data_center
                df_aug[ip]=dics
                testlist=row["testlist"]
                df_test[ip]=process_test(testlist)
                df_center[ip]=dics2
    get_graph_data(df_aug,df_test,df_center,path_bound,outpath)
def re_get_test(path,path_bound,outpath,iplist,max_distance,region):
    df = pd.read_csv(path)
    df_aug={}
    df_test={}
    df_center={}
    for index, row in df.iterrows():
            ip = row["IP"]
            if ip in iplist:
                geolist = row["geolist"]
                geolist = geolist.split("_")
                dics={}
                dics2={}
                for geo in geolist:  # 遍历每一天
                    geo = geo.split(":")
                    date = geo[0] # 每一天
                    date = datelist.index(date)
                    data = geo[1]  # 数据
                    data, data_center = process_geolist(data)  # 获取n天的点和中心
                    points_data = data_augmentation(data, max_distance)
                    dics[str(date)]=points_data
                    dics2[str(date)] = data_center
                df_aug[ip]=dics
                testlist=row["testlist"]
                df_test[ip]=process_test(testlist)
                df_center[ip]=dics2
    get_graph_data(df_aug,df_test,df_center,path_bound,outpath)
def get_graph_data(df_aug,df_test,df_center,path_bound,outpath):
    for f in os.listdir(path_bound):
            ip = ".".join(f.split(".")[:4])
            if ip in df_aug:
                datelist=list(df_aug[ip].keys())
                combination = [[datelist[-7], datelist[-6], datelist[-5], datelist[-4], datelist[-3], datelist[-2], datelist[-1]]]
                df_bound = pd.read_csv(path_bound + "/" + f, header=0)
                bound = get_bound(df_bound.iloc[0]["boundary"])
                geodic = {}  # 初始化geohash字典对应的经纬度
                for geo in bound:
                    lat, lng = geohash_decode(geo)
                    geodic[geo] = [lat, lng]
                lat_mean = df_bound.iloc[0]["lat_mean"]
                lng_mean = df_bound.iloc[0]["lng_mean"]
                lat_diff_mean = df_bound.iloc[0]["lat_diff_mean"]
                lng_diff_mean = df_bound.iloc[0]["lng_diff_mean"]
                train_list = []
                label_list = []
                padding_mask_list = []
                current_label_list=[]
                for comb in combination:
                        train = []
                        label = []
                        padding_mask = []
                        current_label=[]
                        for index in comb:
                            aug = df_aug[ip][index]
                            graph = get_graph(bound,geodic, aug, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
                            latc = graph[0]
                            lngc = graph[1]
                            x = graph[2]
                            edge_index=graph[3]
                            edge_attr=graph[4]
                            x = torch.tensor(x , dtype=torch.float32)  # 使用经纬度作为节点特征
                            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                            edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)
                            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                            train.append(data)
                            current_label.append([latc, lngc])
                        label.append(df_center[ip][comb[-1]])
                        padding_mask.append(0)
                        for index in range(0, 8):  #
                                if str(index) in df_test[ip].keys():
                                    label.append(df_test[ip][str(index)])
                                    padding_mask.append(0)
                                else:
                                    label.append([0,0])
                                    padding_mask.append(1)
                        train_list.append(train)
                        label_list.append(torch.tensor(np.array(label), dtype=torch.float32))
                        padding_mask_list.append(torch.tensor(np.array(padding_mask), dtype=torch.bool))
                        current_label_list.append(torch.tensor(np.array(current_label), dtype=torch.float32))
                print(f"{ip} done!")
                #print(f"train_list:{train_list}, label_list:{label_list}, adding_mask_list:{padding_mask_list},current_label_list:{current_label_list}")
                torch.save((train_list, label_list, padding_mask_list,current_label_list), outpath+"/" + ip + ".pt")


def check_graph_edges(outpath):
    # 记录哪些 IP 的图的边数量为 0
    ip_list_with_no_edges = []

    # 遍历保存的所有 .pt 文件
    for f in os.listdir(outpath):
        ip = ".".join(f.split(".")[0:4])
        file_path = os.path.join(outpath, f)
        # 加载保存的图数据
        train_list, label_list, padding_mask_list, current_label_list = torch.load(file_path)
        # 检查每个图的数据
        for data_list in train_list:
            for data in data_list:
                # 检查边的数量，如果没有边（即 edge_index 的形状第二维为 0）
                if data.edge_index.dim() == 1:
                    ip_list_with_no_edges.append(ip)
                    print(f"IP {ip} has a graph with 0 edges.")
                    break  # 只需要记录一次即可，不用重复检查
    return ip_list_with_no_edges

regions = ["company", "edu", "cstnet"]
for region in regions:
    path=f'data/{region}/{region}.csv'
    path_bound = f'data/{region}/bound_data'
    outpath = f'data/{region}/test_graph_data'
    ip_list_with_no_edges=check_graph_edges(outpath)
    print(ip_list_with_no_edges)
    mindis = 0.05
    maxdis = 0.2
    while len(ip_list_with_no_edges)>0:
        for ip in ip_list_with_no_edges:
            print(outpath+"/"+ip)
        max_distance = random.uniform(mindis,maxdis)
        re_get_test(path, path_bound, outpath, ip_list_with_no_edges,max_distance,region)
        ip_list_with_no_edges = check_graph_edges(outpath)
        mindis=mindis+0.1
        maxdis=maxdis+0.1

    #get_test(path,path_bound,outpath,region)



