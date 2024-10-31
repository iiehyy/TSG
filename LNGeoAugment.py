import math
import numpy as np
import torch
from torch_geometric.data import Data,DataLoader,Dataset,Batch
import pandas as pd
import random
import multiprocessing
import itertools
from scipy.stats import truncnorm
import os
from sklearn.metrics.pairwise import haversine_distances
from datetime import datetime

Base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
datelist = ["20230420","20230421","20230422","20230423","20230424","20230425","20230426","20230427","20230428","20230429","20230430","20230501",
                     "20230502","20230503","20230504","20230505","20230506","20230507","20230508","20230509","20230510","20230511",
                     "20230512","20230513","20230514","20230515","20230516","20230517","20230518","20230519"]
def haversine_distance(point1, point2):
    # 转换为弧度制
    point1 = np.radians(point1)
    point2 = np.radians(point2)
    # 计算距离
    dist = haversine_distances([point1, point2])
    # 将距离转换为地球上的长度
    dist = dist * 6378.137  # 地球半径（k米）
    return dist[1][0]

def geohash_decode(code):
        Base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
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

#数据增强并按照IP存储,在原有的经纬度下，固定最大偏移距离，在8个方向随机偏移，按照高斯分布
def process_geolist(geo8list,region):
    l1 = geo8list.split("_")
    dic = {}
    dic2 = {}
    for i in l1:
        l2 = i.split(":")
        date = l2[0]
        date = datelist.index(date)
        geos = l2[1].split("@")
        geol = []
        lats = []
        lngs = []
        for geo in geos:
            lat, lng = geohash_decode(geo)
            lats.append(lat)
            lngs.append(lng)
            geol.append([lat, lng, date])
        dic[date] = geol
        dic2[date] = [np.mean(lats), np.mean(lngs)]
    return dic, dic2
def generate_truncated_normal(mean, std, min_value, max_value, size):
    samples = truncnorm.rvs(min_value, max_value, loc=mean, scale=std, size=size)
    return samples
def add_gaussian_noise(latitude, longitude, max_distance):
    # 将样本值限制在指定范围内
    # 将最大距离转换为经纬度的差值
    lat_offset = max_distance / 111
    lon_offset = max_distance / (111 * math.cos(math.radians(latitude)))
    valid = False
    directions = {
        'N': (1, 0),
        'NE': (1, 1),
        'E': (0, 1),
        'SE': (-1, 1),
        'S': (-1, 0),
        'SW': (-1, -1),
        'W': (0, -1),
        'NW': (1, -1)
    }
    choice = random.choice(['N','NE','E','SE','S','SW','W','NW'])
    while not valid:
        std_lat = random.uniform(0, 2 * lat_offset)
        mean_lat = random.uniform(0, lat_offset)
        std_lon = random.uniform(0, 2 * lon_offset)
        mean_lon = random.uniform(0, lon_offset)
        # 生成满足条件的随机偏移量
        # 生成高斯分布的随机偏移量
        lat_noise = generate_truncated_normal(mean_lat, std_lat,-lat_offset, lat_offset, 1) #random.gauss(0, lat_offset)
        lon_noise = generate_truncated_normal(mean_lon, std_lon,- lon_offset, lon_offset , 1)#random.gauss(0, lon_offset)
        noisy_latitude = latitude + directions[choice][0]*lat_noise
        noisy_longitude = longitude + directions[choice][1]*lon_noise
        # 计算添加噪声后的距离
        distance = haversine_distance([latitude, longitude],[noisy_latitude[0], noisy_longitude[0]])
        # 判断距离是否满足条件
        if distance <= max_distance:
                valid = True
    return float(noisy_latitude), float(noisy_longitude)
def data_augmentation(data,max_distance):
    augmented_data = []
    for elem in data:
        lat=elem[0]
        lon=elem[1]
        nums=random.randint(5,10)
        for i in range(nums):
            noisy_latitude, noisy_longitude=add_gaussian_noise(lat, lon, max_distance)
            augmented_data.append([noisy_latitude, noisy_longitude])
            augmented_data.append([lat, lon])
    return augmented_data
def augment_data_for_ip(ip, ip_df,region,output_path,mindis,maxdis):
    for index, row in ip_df.iterrows():
        geolist = row["geolist"]
        data, data_center = process_geolist(geolist,region)  # 获取n天的点和中心
        N = 0  # 增强次数
        date_dic = {}
        for date in data.keys():  # 遍历每一天
            #if date not in date_dic.keys():
                date_dic[date] = []
        while N < 100:
            max_distance = random.uniform(mindis, maxdis)
            for date in data.keys():
                points = data[date].copy()
                points_data = data_augmentation(points, max_distance)
                poslist = []
                for l in points_data:
                    lat, lng = l[0], l[1]
                    poslist.append(f"{lat}@{lng}")
                date_dic[date].append("_".join(poslist))
            N += 1
        newdf = pd.DataFrame.from_dict(date_dic)
        out_path = os.path.join(output_path, f"{ip}.csv")
        newdf.to_csv(out_path, header=True)
        print(f"Augment_Processed {ip}")
        return  0
