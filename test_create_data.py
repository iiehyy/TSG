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
datelist2=["20230609", "20230610", "20230611", "20230612", "20230613", "20230614", "20230615",
"20230616", "20230617", "20230618", "20230619", "20230620", "20230621", "20230622",
"20230623", "20230624", "20230625", "20230626", "20230627", "20230628", "20230629",
"20230630", "20230701", "20230702", "20230703", "20230704", "20230705", "20230706",
"20230707", "20230708", "20230709"]

date_dic = {"edu": ['20230803', '20230804', '20230805', '20230806', '20230807', '20230808', '20230809', '20230810',
                    '20230811', '20230812', '20230813', '20230814', '20230815', '20230816', '20230817', '20230818',
                    '20230819', '20230820', '20230821', '20230822', '20230823', '20230824', '20230825', '20230826'],
            "company": ['20230801', '20230802', '20230803', '20230804', '20230805', '20230806', '20230807', '20230808',
                        '20230809', '20230810', '20230811', '20230812', '20230813', '20230814', '20230815', '20230816',
                        '20230817', '20230818', '20230819', '20230820', '20230821', '20230822', '20230823', '20230824'],
            "cstnet": ['20230804', '20230805', '20230806', '20230807', '20230808', '20230809', '20230810', '20230811',
                       '20230812', '20230813', '20230814', '20230815', '20230816', '20230817', '20230818', '20230819',
                       '20230820', '20230821', '20230822', '20230823', '20230824', '20230825', '20230826', '20230827']}
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
test_datelist2=["20230710","20230711","20230712","20230713","20230714","20230715","20230716"]
test_date_dic={"edu":["20230827","20230828","20230829","20230830","20230831","20230901","20230902"],
                "company":['20230825', '20230826', '20230827', '20230828', '20230829', '20230830', '20230831'],
               "cstnet":['20230828', '20230829', '20230830', '20230831', '20230901', '20230902', '20230903']
               }

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
def comparedate(date_str1,date_str2):
    # 日期字符串
    # 将字符串转换为日期对象
    date1 = datetime.strptime(date_str1, "%Y%m%d").date()
    date2 = datetime.strptime(date_str2, "%Y%m%d").date()
    # 比较两个日期
    if date1 > date2:
        return True
    else:
        return False

def process_test(s):
    s=s.split("_")
    dics={}
    for item in s:
        l=item.split(":")
        date=l[0]
        date = test_date_dic[region].index(date)
        '''
        if comparedate(date, "20230709"):
            date = test_datelist2.index(date)
        else:
            date = test_datelist.index(date)
        '''
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
                    date = date_dic[region].index(date)
                    '''
                    if comparedate(date,"20230608"):
                        date = datelist2.index(date)
                    else:
                        date = datelist.index(date)
                    '''
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
                    date = date_dic[region].index(date)
                    '''
                    if comparedate(date,"20230608"):
                        date = datelist2.index(date)
                    else:
                        date = datelist.index(date)
                    '''
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

        if ip != "61.143.53.34":
            continue

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
#regions = ["guangdong", "shaanxi", "hubei"]
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



