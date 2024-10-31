import random
import torch
from torch_geometric.data import Data,DataLoader,Dataset,Batch
class CustomDataset:
    def __init__(self, graphdics, current_label_list, loaded_label_list, loaded_mask_list):
        self.graphdics = graphdics
        self.current_label_list = current_label_list
        self.loaded_label_list = loaded_label_list
        self.loaded_mask_list = loaded_mask_list
    def __len__(self):
        return len(self.graphdics[0])
    def __getitem__(self, index):
        graphdic = {k:v[index] for k,v in self.graphdics.items()}
        current_label = self.current_label_list[index]
        loaded_label = self.loaded_label_list[index]
        loaded_mask = self.loaded_mask_list[index]
        return graphdic, current_label, loaded_label, loaded_mask

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0

    def __iter__(self):
        if self.shuffle:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            self.dataset = [self.dataset[i] for i in indices]
        else:
            indices = list(range(len(self.dataset)))
            self.dataset = [self.dataset[i] for i in indices]
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration

        batch_data = self.dataset[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        graphdics={}
        current_labels_list=[]
        loaded_labels_list=[]
        loaded_masks_list=[]
        for b in batch_data:
            graphdic, current_label, loaded_label, loaded_mask=b
            for key in graphdic.keys():
                if key not in graphdics.keys():
                    graphdics[key]=[]
                graphdics[key].append(graphdic[key])
            current_labels_list.append(current_label)
            loaded_labels_list.append(loaded_label)
            loaded_masks_list.append(loaded_mask)
        for key in graphdics.keys():
            graphdics[key]=Batch.from_data_list(graphdics[key])
        current_labels=torch.stack(current_labels_list,0)
        loaded_labels=torch.stack(loaded_labels_list,0)
        loaded_masks=torch.stack(loaded_masks_list,0)
        return graphdics,current_labels,loaded_labels,loaded_masks

    def __len__(self):
        # 返回总共的批次数量
        return len(self.dataset)



def get_test_dataloader(path, batch_size,days):
    loaded_data_list,loaded_label_list,loaded_mask_list,current_label_list= torch.load(path)#n*14*tu,n*8*2,n*8
    graphdics={}
    geo_point_list_train_dics={}
    for day in range(days):
        graphdics[day]=[]
        geo_point_list_train_dics[day]=[]
    for  i in range(len(loaded_data_list)):
        for j in range(len(loaded_data_list[i])):
            graphdics[j].append(loaded_data_list[i][j])
    # 示例使用
    # 假设有一个包含7天图数据的字典以及相应的label列表和mask列表
    # 创建自定义数据集
    custom_dataset = CustomDataset(graphdics, current_label_list, loaded_label_list, loaded_mask_list)
    custom_dataloader = CustomDataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    return custom_dataloader
