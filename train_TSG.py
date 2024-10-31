import torch
import torch.nn as nn
from torch.optim import Adam
import torch.multiprocessing as mp
from lib.GeoGraph import decode_scale
from lib.Dataloader import get_dataloader
from lib.Test_Dataloader import get_test_dataloader
from lib import TSG
import numpy as np
import pandas as pd
import os
import time
import argparse
from multiprocessing import Event, Process, Queue
from additional_compare_experience.TSG.lib.TSG import TSG
from additional_compare_experience.TSG.lib.STGAT import ST_GAT
from additional_compare_experience.TSG.lib.STGCNN import social_stgcnn



# 定义 Transformer 模型
def map_distance(a, b):
    return 6378.137 * 2 * np.arcsin(np.sqrt(
        np.power(np.sin((a[0] - b[0]) * np.arccos(-1) / 360), 2) + np.cos(a[0] * np.arccos(-1) / 180) * np.cos(
            b[0] * np.arccos(-1) / 180) * np.power(np.sin((a[1] - b[1]) * np.arccos(-1) / 360), 2)))
def min_distance(user_loc, pois):
    distance = map_distance(pois, user_loc)
    return np.mean(distance)
def _dist(p, q):  # 经纬度距离聚类
    return min_distance(p, q)
def check(ip,modelpath,path_model_manage):
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    checkpoint = torch.load(modelpath,map_location='cpu')
    obj = checkpoint["s_epoch"]
    obj=obj.split("@")
    obj=obj[-1].split(":")[0]
    if obj=="40" or obj=="50":
        with open(path_model_manage, "a") as file:
                file.write(ip + "\n")
def error(pred, true, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean):
    pred = pred.tolist()
    true = true.tolist()
    dis = 0
    c = 0
    for i in range(len(pred)):
        lat1 = pred[i][0]
        lng1 = pred[i][1]

        lat2 = true[i][0]
        lng2 = true[i][1]

        # print(lat1,lat2,lng1,lng2)
        if lat2 != 0 and lng2 != 0:
            lat1, lng1 = decode_scale(lat1, lng1, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
            lat2, lng2 = decode_scale(lat2, lng2, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
            dis += _dist([lat1, lng1], [lat2, lng2])
            c = c + 1
    return dis / c
from sklearn.metrics import mean_squared_error, mean_absolute_error
def calculate_metrics(errors):
    rmse = np.sqrt(mean_squared_error([0] * len(errors), errors))
    mae = mean_absolute_error([0] * len(errors), errors)
    median_error = np.median(errors)
    return rmse, mae, median_error
def flatten_output(output):
    # 如果是嵌套列表，则展平
    if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
        return output[0]
    return output
def error_test(pred, true, lat_mean, lng_mean, lat_diff_mean, lng_diff_mean):
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
# 定义位置编码和生成掩码的函数（参考前面的代码）
def compute_accuracy(predictions, labels):
    m = nn.Softmax(dim=2)
    predictions = m(predictions)
    _, predicted_labels = torch.max(predictions, dim=2)
    correct = (predicted_labels == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total
    return accuracy
# 创建图数据和目标数据（参考前面的代码）
def cut_feature(data_graph,cut_list):
    data=data_graph.x
    indices = [i for i in range(data.shape[1]) if i not in cut_list]

    # 使用 index_select 函数选择特定的特征维度
    selected_features = data.index_select(1, torch.tensor(indices))
    data_graph.x=selected_features
    return data_graph
def TSG_Init():
    # 定义超参数
    learning_rate = 3e-4
    edge_dim = 1
    in_features = 17
    hidden_features = 512
    out_features = 256
    dim_feedforward = 1024
    num_heads = 2#dim_feedforward%num_heads
    num_layers = 2
    model = TSG(edge_dim, in_features, hidden_features, out_features, dim_feedforward, num_heads, num_layers)
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=5e-5)
    return model,optimizer
def ST_GAT_Init(N_DAY_SLOT):
    # 定义超参数
    config = {
        'BATCH_SIZE': 50,
        'EPOCHS': 200,
        'WEIGHT_DECAY': 5e-5,
        'INITIAL_LR': 3e-4,
        'N_PRED': 9,
        'N_HIST': 7,
        'DROPOUT': 0.2,
        # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
        'USE_GAT_WEIGHTS': True,
        'N_NODE':N_DAY_SLOT,
    }
    # Number of possible windows in a day
    model = ST_GAT(in_channels=config['N_HIST'], out_channels=config['N_PRED'], n_nodes=config['N_NODE'],dropout=config['DROPOUT'])
    optimizer = Adam(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    return model,optimizer
def social_stgcnn_Init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
    args = parser.parse_args()
    # 定义超参数
    model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len)
    return model

def train(ip, path_model, path_boundary, path_graph, path_model_manage, gpu_id, task_complete,models,test_path_graph):
            # 设置要使用的 GPU 编号
            gpu_id = str(gpu_id)  # 或者设置为其他 GPU 编号，如 1、2 等
            device = torch.device("cuda:"+gpu_id)
            torch.cuda.set_device(device)
            # 清空选定GPU设备的内存
            torch.cuda.empty_cache()
            torch.autograd.set_detect_anomaly(True)
            batch_size = 32
            num_epochs = 41
            daylength = 7
            # 创建数据加载器
            criterion = nn.MSELoss()  # 定义损失函数和优化器
            s_epoch = ""
            s_error = ""
            boundarydata = path_boundary+"/" + ip + ".csv"
            test_graphdata = test_path_graph + "/" + ip + ".pt"
            graphdata = path_graph+"/" + ip + ".pt"
            ipbound = pd.read_csv(boundarydata, index_col=None)
            ipbound = ipbound.iloc[0]
            lat_mean, lng_mean, lat_diff_mean, lng_diff_mean = ipbound["lat_mean"], ipbound["lng_mean"], ipbound["lat_diff_mean"], ipbound["lng_diff_mean"]
            dataloader,geolist = get_dataloader(graphdata, batch_size, daylength)
            test_dataloader = get_test_dataloader(test_graphdata, 1, daylength)
            if models=="TSG":
                model,optimizer = TSG_Init()  # TransformerModel(num_features, hidden_dim, num_heads, num_layers)# 创建模型
            if models == "ST_GAT":
                model,optimizer = ST_GAT_Init(len(geolist))
            if models == "social_stgcnn":
                model,optimizer = social_stgcnn_Init()
            # 假设您的模型名为 SimpleModel
            model = model.to(device)
            # 打印每一层的参数形状
            step = 0
            start_time=time.time()
            loss=0
            min_median_error = float('inf')  # 初始化为无穷大
            for epoch in range(num_epochs):
                    model.train()
                    if epoch>10 and loss < 10:
                        learning_rate = 0.001
                        optimizer = Adam(model.parameters(), lr=learning_rate)
                    for graphdics, current_labels, loaded_labels, loaded_masks,geo_point_list_train_dic in dataloader:
                            data_list=[]
                            for day in graphdics:
                                data=graphdics[day]
                                data_list.append(data.to(device))
                            optimizer.zero_grad()
                            targets = loaded_labels.to(device)
                            current_label = current_labels.to(device)
                            out_pred = model(data_list,current_label)
                            loss = criterion(out_pred, targets[:, 1, :])
                            loss.backward()
                            optimizer.step()
                            pred_error = error(out_pred, targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean,
                                               lng_diff_mean)  # 先经纬度解码再计算预测和后一天定位误差
                            base_error = error(targets[:, 0, :], targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean,
                                               lng_diff_mean)  # 先经纬度解码再计算今天和后一天定位误差
                            print(ip + "_loss:", loss)
                            print(ip + "_pred_error", pred_error)
                            print(ip + "_base_error", base_error)

                            '''
                            if step % 50 == 0:
                                pred_error = error(out_pred, targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)#先经纬度解码再计算预测和后一天定位误差
                                base_error = error(targets[:, 0, :], targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean,lng_diff_mean)#先经纬度解码再计算今天和后一天定位误差
                                print(ip+"_loss:", loss)
                                print(ip+"_pred_error", pred_error)
                                print(ip+"_base_error", base_error)
                            '''
                            step = step + 1
                    pred_error = error(out_pred, targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean, lng_diff_mean)
                    loss_v = str(loss.cpu().item())
                    s_epoch = s_epoch + "@" + str(epoch) + ":" + loss_v
                    s_error = s_error + "@" + str(epoch) + ":" + str(pred_error)
                    if epoch % 10 == 0:
                        end_time = time.time()
                        run_time = str(end_time - start_time)
                        print(ip + f"_Epoch {epoch + 1} - Loss: {loss}")
                        basic_save_info = {
                            'step': step,
                            'ip': ip,
                            'loss': loss,  # 确保是Python原始类型
                            's_epoch': s_epoch,
                            's_error': s_error,
                            'run_time': run_time
                        }
                        #torch.save(basic_save_info,  path_model + "/" + ip + "_modle")
                        model.eval()  # 切换到评估模式
                        with torch.no_grad():  # 不需要计算梯度
                            error_List=[]
                            for test_graphdics, test_current_labels, test_loaded_labels, test_loaded_masks in test_dataloader:
                                test_data_list = []
                                for day in test_graphdics:
                                    test_data = test_graphdics[day]
                                    test_data_list.append(test_data.to(device))
                                test_targets = test_loaded_labels.to(device)
                                test_current_label = test_current_labels.to(device)
                                out_pred = model(test_data_list, test_current_label)
                                pred_error = error_test(out_pred, test_targets[:, 1, :], lat_mean, lng_mean, lat_diff_mean,
                                                   lng_diff_mean)  # 先经纬度解码再计算预测和后一天定位误差
                                error_List.append(pred_error)
                            rmse,mae,median_error=calculate_metrics(error_List)
                            if median_error < min_median_error:
                                min_median_error = median_error
                                full_save_info = {
                                    **basic_save_info,  # 包含基本信息
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()
                                }
                                #torch.save(full_save_info, path_model + "/" + ip + "_modle_best")
                            print(f"RMSE: {rmse}, MAE: {mae}, Median Error: {median_error}")
            check(ip,path_model+"/" + ip + "_modle", path_model_manage)
            task_complete.set()
            return 0

def read_iplist_from_file(path):
        # 从文件读取训练完的IP
        with open(path, "r") as file:
            lines = file.readlines()
            iplist = [line.strip() for line in lines]
        return iplist
def manage_tasks(gpu_ids, ip_queue, path_model, path_boundary, path_graph, path_model_manage, ip_per_gpu,models,test_path_graph):
    events = {gpu_id: [Event() for _ in range(ip_per_gpu[gpu_id])] for gpu_id in gpu_ids}
    processes = {gpu_id: [None for _ in range(ip_per_gpu[gpu_id])] for gpu_id in gpu_ids}
    print(ip_queue)
    # 初始分配任务
    for gpu_id in gpu_ids:
        for i in range(ip_per_gpu[gpu_id]):
            if not ip_queue.empty():
                ip = ip_queue.get()
                events[gpu_id][i].clear()  # 清空事件状态
                process = Process(target=train, args=(ip, path_model, path_boundary, path_graph, path_model_manage, gpu_id, events[gpu_id][i],models,test_path_graph))
                processes[gpu_id][i] = process
                process.start()

    # 循环检查事件状态并重新分配任务
    active = True
    while active:
        active = False
        for gpu_id in gpu_ids:
            for i in range(ip_per_gpu[gpu_id]):
                if events[gpu_id][i].is_set():
                    if processes[gpu_id][i] is not None:
                        processes[gpu_id][i].join()  # 确保进程已经结束
                    if not ip_queue.empty():
                        ip = ip_queue.get()
                        events[gpu_id][i].clear()
                        process = Process(target=train, args=(ip, path_model, path_boundary, path_graph, path_model_manage, gpu_id, events[gpu_id][i],models,test_path_graph))
                        processes[gpu_id][i] = process
                        process.start()
                    else:
                        processes[gpu_id][i] = None  # 没有更多的 IP，设置为 None
                if processes[gpu_id][i] is not None:
                    active = True  # 至少有一个 GPU 仍在处理任务

    # 确保所有 GPU 上的所有进程都已完成
    for gpu_id in gpu_ids:
        for process in processes[gpu_id]:
            if process is not None:
                process.join()

    print("所有任务已完成。")

if __name__ == '__main__':
    # Setup paths and GPU information
    mp.set_start_method('spawn')  # 对于 PyTorch 和 CUDA 操作，通常需要 'spawn' 或 'forkserver'
    #regions = ["guangdong", "shaanxi", "hubei"]
    #regions = ["edu","company","cstnet","guangdong", "shaanxi", "hubei"]
    regions=["guangdong"]
    models = "TSG"
    num_gpus = 4
    ip_per_gpu = [2,0,2,0]  # 根据 GPU 的能力预先设定每个 GPU 可以处理的最大 IP 数量
    # Initialize paths and queues
    path_model_base = "model"
    path_data_base = "data"
    for region in regions:
        df=pd.read_csv(f"data/{region}/{region}.csv")
        iplist_t=df["IP"].tolist()
        path_model_manage = os.path.join(path_model_base, region, "model_manage", f"{models}_iplist.txt")
        path_model = os.path.join(path_model_base, region, f"model_{models}")
        path_aug = os.path.join(path_data_base, region, "augment_data")
        path_boundary = os.path.join(path_data_base, region, "bound_data")
        path_graph = os.path.join(path_data_base, region, "graph_data")
        test_path_graph = os.path.join(path_data_base, region, "test_graph_data")
        if not os.path.exists(path_model):
            os.makedirs(path_model)
        if not os.path.isfile(path_model_manage):
            with open(path_model_manage, 'w') as file:
                file.write('')
        exit_ip = read_iplist_from_file(path_model_manage)
        #iplist_t = [".".join(f.split(".")[0:4]) for f in os.listdir(path_aug) if os.path.isfile(os.path.join(path_aug, f))]
        iplist = [ip for ip in iplist_t if ip not in exit_ip]
        print(len(exit_ip),len(iplist))
        iplist =["113.71.145.112","113.117.3.131","113.72.218.45","113.77.48.68","113.77.48.117","113.109.196.3"]
        ip_queue = Queue()
        for ip in iplist:
            ip_queue.put(ip)
        # Start managing tasks
        manage_tasks(list(range(num_gpus)), ip_queue, path_model, path_boundary, path_graph, path_model_manage, ip_per_gpu,models,test_path_graph)
