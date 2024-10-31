# -*- coding: utf-8 -*-
from lib import GeoGraph
from lib import GeoHashMap
from lib import LNGeoAugment
import os
import multiprocessing
import pandas as pd
from lib import Dataloader
import torch
from torch_geometric.data import Data,DataLoader,Dataset,Batch
from lib.Dataloader import get_dataloader
def validate_graph(path,days):
    valid_ips=[]
    for f in os.listdir(path):
        try:
            ip = ".".join(f.split('.')[:-1])
            #try:
            dataloader,geolist=Dataloader.get_dataloader(path+"/"+f,1, days)
            lenth=len(dataloader)
            if lenth >2000:
                valid_ips.append(ip)
        except Exception as e:
            print(f"Error processing {ip}")
    return valid_ips
def validate_files(path, required_lines=100, check_lines=True):
    """通用文件验证函数，根据阶段需求验证文件行数或大小"""
    valid_ips = []
    for f in os.listdir(path):

        ip = ".".join(f.split('.')[:-1])
        full_path = os.path.join(path, f)

        try:
            # 如果需要检查行数，读取文件行数；否则仅检查文件大小
            if check_lines:
                with open(full_path, 'r') as file:
                    lines = file.readlines()
                    if len(lines) > required_lines:
                        valid_ips.append(ip)
            else:
                if os.path.getsize(full_path) > 0:
                    valid_ips.append(ip)

        except Exception as e:
            print(f"Error processing file {full_path}: {e}")

    return valid_ips


def process_augment(region):
    inputpath = f'data/{region}/{region}.csv'
    outpath = f'data/{region}/augment_data'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    df = pd.read_csv(inputpath)
    ips = set(df['IP'].unique())  # 所有的独特IP地址
    # 分别获取已经有效处理的IP和需要重新处理的IP
    valid_ips= validate_files(outpath)
    ips_to_process = ips - set(valid_ips)
    print("augment",region,len(valid_ips),len(ips_to_process))
    results = []
    pool = multiprocessing.Pool(processes=20)
    for ip in ips_to_process:
        result=pool.apply_async(LNGeoAugment.augment_data_for_ip, args=(ip, df[df['IP'] == ip],region,outpath))
        results.append(result)
    pool.close()
    pool.join()
    # 确保所有任务都成功完成
    for result in results:
        if not result.successful():
            print("Some tasks did not complete successfully")

def process_bound(region):
    inputpath = f'data/{region}/augment_data'
    outpath = f'data/{region}/bound_data'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    valid_aug_ip = validate_files(inputpath)#要处理的全集
    valid_bound_ips = validate_files(outpath,check_lines=False)
    # 确定哪些有效的增强文件尚未进行bound处理或需要重新处理
    process_ip=set(valid_aug_ip)-set(valid_bound_ips)
    print("bound",region, len(valid_bound_ips), len(process_ip))
    results = []
    pool = multiprocessing.Pool(processes=20)
    for ip in process_ip:
        result = pool.apply_async(GeoHashMap.get_boundary, args=(ip,inputpath,outpath))
        results.append(result)
    pool.close()
    pool.join()

    # 确保所有任务都成功完成
    for result in results:
        if not result.successful():
            print("Some tasks did not complete successfully")
            
def process_graph(region):
    date_interval=7
    path_aug='data/'+region+'/augment_data'
    path_bound = 'data/' + region + '/bound_data'
    outpath = f'data/{region}/graph_data'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    valid_bound_ips = validate_files(path_bound,check_lines=False)
    valid_graph_ips = validate_graph(outpath,7)
    process_ip=set(valid_bound_ips)-set(valid_graph_ips)
    print("graph",region,len(valid_bound_ips),len(process_ip))
    results = []
    pool = multiprocessing.Pool(processes=20)
    for ip in process_ip:
        result = pool.apply_async(GeoGraph.get_Graph, args=(path_aug,path_bound,outpath,ip,date_interval))
        results.append(result)
    pool.close()
    pool.join()
    # 确保所有任务都成功完成
    for result in results:
        if not result.successful():
            print("Some tasks did not complete successfully")

def check_graph_edges(outpath):
    # 记录哪些 IP 的图的边数量为 0
    ip_list_with_no_edges = []

    # 遍历保存的所有 .pt 文件
    for f in os.listdir(outpath):
        ip = ".".join(f.split(".")[0:4])
        file_path = os.path.join(outpath, f)
        c=0
        # 加载保存的图数据
        train_list, label_list, padding_mask_list, current_label_list, geolist, geo_point_list_train_list= torch.load(file_path)
        # 检查每个图的数据
        for data_list in train_list:
            for data in data_list:
                # 检查边的数量，如果没有边（即 edge_index 的形状第二维为 0）
                if data.edge_index.dim() == 1:
                    c=c+1
                    break  # 只需要记录一次即可，不用重复检查
            if c>800:
                ip_list_with_no_edges.append(ip)
                print(f"IP {ip} has a graph with many 0 edges.")
                break
    return list(set(ip_list_with_no_edges))


def process_ip(ip, df, path_aug, region, path_bound, outpath, date_interval,mindis,maxdis):
        # 对每个IP执行三个步骤
        LNGeoAugment.augment_data_for_ip(ip, df[df['IP'] == ip], region, path_aug,mindis,maxdis)
        GeoHashMap.get_boundary(ip, path_aug, path_bound)
        GeoGraph.get_Graph(path_aug, path_bound, outpath, ip, date_interval)
        return 0
def re_create(path, path_aug, path_bound, outpath, iplist,region,mindis,maxdis):
    date_interval = 7
    df = pd.read_csv(path)
    pool =multiprocessing.Pool(processes=20)  # 使用20个进程
    results = []
    # 定义一个包装函数来顺序执行每个IP的三个任务
    # 异步启动所有任务
    for ip in iplist:
        result = pool.apply_async(process_ip, (ip, df, path_aug, region, path_bound, outpath, date_interval,mindis,maxdis))
        results.append(result)
    outputs = [result.get() for result in results]

    # 关闭池并等待所有进程结束
    pool.close()
    pool.join()

    return outputs  # 返回从每个任务中收集的结果

def re_process():
    regions = ["company","cstnet","edu"]
    for region in regions:
        mindis = 0.05
        maxdis = 0.2
        path = f'data/{region}/{region}.csv'
        path_aug = 'data/' + region + '/augment_data'
        path_bound = 'data/' + region + '/bound_data'
        outpath = f'data/{region}/graph_data'
        ip_list_with_no_edges = check_graph_edges(outpath)
        print(region,len(ip_list_with_no_edges))
        while len(ip_list_with_no_edges) > 0:
            for ip in ip_list_with_no_edges:
                print(outpath + "/" + ip)
            mindis=mindis+0.1
            maxdis=maxdis+0.2
            re_create(path,path_aug,path_bound, outpath, ip_list_with_no_edges,region,mindis,maxdis)
            ip_list_with_no_edges = check_graph_edges(outpath)
            print(region, len(ip_list_with_no_edges))

if __name__ == "__main__":
    regions = ["company", "cstnet","edu"]
    for region in regions:
        process_augment(region)
        process_bound(region)
        process_graph(region)
