# -*- coding: utf-8 -*-
from torch.optim import Adam,SGD
import torch
from lib.GeoGraph import decode_scale
from lib import TSG
import pandas as pd
import os
import argparse
from additional_compare_experience.TSG.lib.TSG import TSG
from additional_compare_experience.TSG.lib.STGAT import ST_GAT
from additional_compare_experience.TSG.lib.STGCNN import social_stgcnn
import numpy as np
from lib.Test_Dataloader import get_test_dataloader
from sklearn.metrics import mean_squared_error, mean_absolute_error
def TSG_Init():
    # 定义超参数
    learning_rate = 0.01
    edge_dim = 1
    in_features = 17
    hidden_features = 512
    out_features = 256
    dim_feedforward = 1024
    num_heads = 2
    num_layers = 2
    model = TSG(edge_dim, in_features, hidden_features, out_features, dim_feedforward, num_heads, num_layers)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.1)
    return model,optimizer
def ST_GAT_Init(N_DAY_SLOT):
    # 定义超参数
    config = {
        'BATCH_SIZE': 32,#50
        'EPOCHS': 200,
        'WEIGHT_DECAY': 1,#5e-5,
        'INITIAL_LR': 0.1,#3e-4,
        'N_PRED': 3,#9,
        'N_HIST': 7,#天数
        'DROPOUT': 0.2,
        # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
        'USE_GAT_WEIGHTS': True,
        'N_NODE':N_DAY_SLOT,#节点数量
    }
    # Number of possible windows in a day
    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'],heads=2, n_nodes=config['N_NODE'],dropout=config['DROPOUT'])
    optimizer = Adam(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    return model,optimizer
def social_stgcnn_Init(n_node):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=17)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')#1
    parser.add_argument('--n_txpcnn', type=int, default=1, help='Number of TXPCNN layers')#5
    parser.add_argument('--kernel_size', type=int, default=3)#原始为3
    # Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=7)
    parser.add_argument('--pred_seq_len', type=int, default=12)

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')#0.01

    args = parser.parse_args()

    # 定义超参数
    model = social_stgcnn(n_node=n_node,n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,output_feat=args.output_size, seq_len=args.obs_seq_len,kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
    #optimizer = SGD(model.parameters(), lr=args.lr)
    #optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.1)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)
    return model,optimizer


def map_distance(a, b):
    return 6378.137 * 2 * np.arcsin(np.sqrt(
        np.power(np.sin((a[0] - b[0]) * np.arccos(-1) / 360), 2) + np.cos(a[0] * np.arccos(-1) / 180) * np.cos(
            b[0] * np.arccos(-1) /  180) * np.power(np.sin((a[1] - b[1]) * np.arccos(-1) / 360), 2)))
def min_distance(user_loc, pois):
    distance = map_distance(pois, user_loc)
    return np.mean(distance)
def _dist(p, q):  # 经纬度距离聚类
    return min_distance(p, q)
def flatten_output(output):
    # 如果是嵌套列表，则展平
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        return output[0]
    return output
def error(pred, true, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean):
    pred = pred.tolist()
    true = true.tolist()
    pred = flatten_output(pred)
    true = flatten_output(true)
    lat1 = pred[0]
    lng1 = pred[1]
    lat2 = true[0]
    lng2 = true[1]
    lat1, lng1 = decode_scale(lat1, lng1, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
    dis= _dist([lat1, lng1], [lat2, lng2])
    return dis

def get_bound(bound):
    boundlist=bound.split("_")
    return boundlist
def test(ip, path_model, path_boundary, path_graph, gpu_id, models):
    gpu_id = str(gpu_id)  # 或者设置为其他 GPU 编号，如 1、2 等
    device = torch.device("cuda:" + gpu_id)
    torch.cuda.set_device(device)
    # 清空选定GPU设备的内存
    torch.cuda.empty_cache()
    batch_size = 1
    daylength = 7
    # 创建数据加载器
    boundarydata = path_boundary + "/" + ip + ".csv"
    graphdata = path_graph + "/" + ip + ".pt"
    path_model=path_model + "/" + ip+"_modle_best"
    #print(path_model)
    ipbound = pd.read_csv(boundarydata, index_col=None)
    ipbound = ipbound.iloc[0]
    lat_mean, lng_mean, lat_diff_mean, lng_diff_mean = ipbound["lat_mean"], ipbound["lng_mean"], ipbound["lat_diff_mean"], ipbound["lng_diff_mean"]
    dataloader= get_test_dataloader(graphdata, batch_size, daylength)
    geolist=get_bound(ipbound['boundary'])
    checkpoint = torch.load(path_model, map_location=device)
    obj = checkpoint["model_state_dict"]
    if models == "TSG":
        model, _= TSG_Init()
    elif models == "ST_GAT":
        model, _ = ST_GAT_Init(len(geolist))
    else :
        model, _ = social_stgcnn_Init(len(geolist))
    model.load_state_dict(obj)
    model = model.to(device)
    for name, param in model.named_parameters():
        if name=="gat.gat1.lin_l.weight":
            return torch.mean(torch.abs(param),dim=1).tolist()
            #print(ip,torch.mean(torch.abs(param),dim=1).tolist())
        #    print(f"Layer: {name} | Size: {param.size()} | Values : {param}")
        #print(f"Layer: {name} | Size: {param.size()}")# | Values : {param}")
    #print("---------------------------------------")

    model.eval()
    with torch.no_grad():
        for graphdics, current_labels, loaded_labels, loaded_masks in dataloader:
            data_list = []
            for day in graphdics:
                data = graphdics[day]
                data_list.append(data.to(device))
            targets = loaded_labels.to(device)
            current_label = current_labels.to(device)
            out_pred = model(data_list, current_label)
            pred_error = error(out_pred, targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean,lng_diff_mean)  # 先经纬度解码再计算预测和后一天定位误差
            base_error = error(targets[:, 0, :], targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean,lng_diff_mean)  # 先经纬度解码再计算今天和后一天定位误差

    return base_error#pred_error
def get_iplist(modelpath,previous_ipset):
    ipset=set()
    for f in os.listdir(modelpath):
        ip = f.split("_")[0]
        ipset.add(ip)
    # 如果有之前的ipset，做交集
    if previous_ipset is not None:
        ipset = ipset.intersection(previous_ipset)
    return ipset
def calculate_metrics(errors):
    rmse = np.sqrt(mean_squared_error([0] * len(errors), errors))
    mae = mean_absolute_error([0] * len(errors), errors)
    median_error = np.median(errors)
    error_less_than_5km = np.mean(np.array(errors) < 5)
    return (rmse, mae, median_error,error_less_than_5km)
def cal_weight(data):
    data_array = np.array(data)
    # 按列求均值，即每个维度的均值
    mean_vector = list(np.sum(data_array, axis=0))

    Central_Location_Descriptor=(mean_vector[0]+mean_vector[1])/2
    Geohash_Offset_Descriptor=mean_vector[2]
    Global_Positional_Offset_Descriptor=(mean_vector[3]+mean_vector[4]+mean_vector[5]+mean_vector[6])/4
    Local_Dispersion_Descriptor=(mean_vector[7]+mean_vector[8]+mean_vector[9]+mean_vector[10])/4
    Local_Aggregation_Descriptor=(mean_vector[13]+mean_vector[14]+mean_vector[15]+mean_vector[16])/4
    Density_Distribution_Descriptor=(mean_vector[11]+mean_vector[12])/2

    return [Central_Location_Descriptor,Geohash_Offset_Descriptor,Global_Positional_Offset_Descriptor,Local_Dispersion_Descriptor,Local_Aggregation_Descriptor,Density_Distribution_Descriptor]
#regions = ["guangdong","hubei","guangdong","cstnet","edu","company"
regions=["guangdong"]#,"hubei","shaanxi","cstnet","edu","company"]guangdong
models=["TSG"]#"TSG",,"social_stgcnn"
gpu_id=3
for region in regions:
    previous_ipset = None
    for model in models:
        path_model = f'model/{region}/model_{model}'
        previous_ipset = get_iplist(path_model, previous_ipset)
    iplist = list(previous_ipset)
    dics={}
    for model in models:
        print(region,model)
        if model not in dics:
            dics[model]=[]
        path=f'data/{region}/{region}.csv'
        path_bound = f'data/{region}/bound_data'
        path_graph = f'data/{region}/test_graph_data'
        path_model = f'model/{region}/model_{model}'
        for ip in iplist:
                erro_dis=test(ip, path_model, path_bound, path_graph, gpu_id, model)#{'ST_GAT': (3.347360233931958, 3.347360233931958, 3.347360233931958), 'social_stgcnn': (0.7874729824445306, 0.7874729824445306, 0.7874729824445306)}
                if ip!="112.96.33.52":
                    dics[model].append(erro_dis)
        #print(len(dics[model]))
        #dics[model]=calculate_metrics(dics[model])
        dics[model] = cal_weight(dics[model])
    print(dics)
'''
#RMSE,MAE,Median Error
{'TSG': (4.999980925581786, 3.284227130613872, 2.6352012299868193), 
 'ST_GAT': (4.548252834444694, 2.8853270859875435, 2.054768014397763), 
 'social_stgcnn': (4.929102225502561, 3.1734303448901207, 2.61477984958362)}
'''