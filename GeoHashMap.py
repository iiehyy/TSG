import  numpy as np
import math
import multiprocessing
import os
import pandas as pd

Base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

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
def get_geohash_neighbors(geohash):
    # 定义邻居方向的偏移量
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

    # 获取给定 Geohash 的长度
    geohash_length = len(geohash)

    # 获取 Geohash 的坐标（纬度，经度）
    lat, lon = geohash_decode(geohash)

    # 存储邻居 Geohash 编码的列表
    neighbors = set()
    factor=600#geohash6
    # 遍历邻居方向
    shifting_lat = (90/ (2 ** (geohash_length * 5))) * factor
    shifting_lng = (180/ (2 ** (geohash_length * 5))) * factor
    for direction in directions.values():
        dx, dy = direction
        # 计算邻居的坐标（纬度，经度）
        neighbor_lat = lat + dx * shifting_lat
        neighbor_lon = lon + dy * shifting_lng
        # 计算邻居的 Geohash 编码
        neighbor_geohash = geohash_encode(neighbor_lat, neighbor_lon, geohash_length)
        i=0
        while neighbor_geohash == geohash or neighbor_geohash in list(neighbors):
            # 增加偏移量，确保获得不同的邻居
            neighbor_lat += dx * shifting_lng
            neighbor_lon += dy * shifting_lat
            neighbor_geohash = geohash_encode(neighbor_lat, neighbor_lon, geohash_length)
            i=i+1
        neighbors.add(neighbor_geohash)
    return neighbors
def get_neighbors(geohash_str, n):#n代表n层邻居
    neighbors = set()
    neighbors.add(geohash_str)
    neighbors_pass=set()
    #lat,lng=geohash_decode(geohash_str)
    for k in range(n):
        new_neighbors = set()
        for neighbor in neighbors:
            if neighbor not in neighbors_pass:
                one_level_neighbors = get_geohash_neighbors(neighbor)
                new_neighbors.update(one_level_neighbors)
                neighbors_pass.add(neighbor)
        neighbors.update(new_neighbors)
    return neighbors
def process_augment_data(s):
    s=str(s)
    locs=set()
    locs2=[]
    s=s.split("_")
    for geo in s:
        geo=geo.split("@")
        lat,lng=float(geo[0]),float(geo[1])
        locs.add(geohash_encode(lat,lng,6))
        locs2.append([lat,lng])
    return locs,locs2
def get_boundary(ip,inputpath,outpath):
        df=pd.read_csv(inputpath+"/"+ip+".csv",index_col=False)
        geohash_set=set()
        boundary = set()
        loc_list=[]
        dates = df.columns.tolist()
        dates=dates[1:]
        for d in dates:
            s=df[d]
            for i in range(len(s)):
                locs,locs2=process_augment_data(s[i])
                geohash_set.update(locs)
                loc_list=loc_list+locs2
        for geo in geohash_set:
                boundary.update(get_neighbors(geo, 2))

        boundary="_".join(list(boundary))
        lat_mean = sum(point[0] for point in loc_list) / len(loc_list)
        # 计算经度的均值
        lng_mean = sum(point[1] for point in loc_list) / len(loc_list)
        lat_diffs = np.diff([point[0] for point in loc_list])
        lng_diffs = np.diff([point[1] for point in loc_list])
        # 计算差值的均值
        lat_diff_mean = np.mean(np.abs(lat_diffs))
        lng_diff_mean = np.mean(np.abs(lng_diffs))
        #获取IP的地理边界
        dic = {"ip": [ip],"boundary": [boundary], "lat_mean": [lat_mean], "lng_mean": [lng_mean],"lat_diff_mean": [lat_diff_mean], "lng_diff_mean": [lng_diff_mean]}

        newdf=pd.DataFrame(dic)
        newdf.to_csv(outpath+"/"+ip+".csv", header=True)
        print(f"Geohashmap_Processed {ip}")
        return  0
