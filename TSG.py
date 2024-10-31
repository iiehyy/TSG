import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_add_pool
import math
from torchsummary import summary
# 定义 Transformer 模型
class GEN(nn.Module):
    def __init__(self, edge_dim, in_features, hidden_features, out_features, num_heads):
        super(GEN, self).__init__()
        self.gat1 = GATv2Conv(in_features, hidden_features, heads=num_heads, edge_dim=edge_dim, dropout=0.2,bias=True)  # 记得加edge_dim，是边属性特征数量
        self.gat2 = GATv2Conv(hidden_features * num_heads, out_features, heads=1, dropout=0.2, bias=True)
        self.relu = nn.ReLU()
        self.spatial_attention = nn.Linear(out_features, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        mask = (x.sum(dim=1) != 0).float().unsqueeze(1)
        if edge_index.numel() == 0:
            random_node = torch.randint(0, x.size(0), (1,)).item()
            edge_index = torch.tensor([[random_node], [random_node]], dtype=torch.long, device=x.device)
            edge_attr = torch.zeros((1, edge_index.size(1)), dtype=edge_attr.dtype, device=edge_attr.device)
        x = self.gat1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.gat2(x, edge_index)
        x=x*mask
        # Global attention mechanism
        batch_size = batch[-1] + 1
        num_nodes_list = [torch.sum(batch == i).item() for i in range(batch_size)]
        start_idx = 0
        xs = []
        for i in range(batch_size):
            end_idx = start_idx + num_nodes_list[i]
            batch_global_weights = self.spatial_attention(x[start_idx:end_idx])  # Compute attention weights for each batch
            mask_batch = (batch_global_weights != 0).float()
            batch_global_weights = batch_global_weights - (1 - mask_batch) * (1000000000000000.0)
            batch_global_weights = self.softmax(batch_global_weights)  # Apply softmax to obtain attention weights
            p = x[start_idx:end_idx] * batch_global_weights  # Apply attention weights to node features
            xs.append(p)
            start_idx = end_idx
        x = torch.cat(xs, dim=0)
        x=global_add_pool(x, batch)
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=30):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=0.1)
        # 计算位置编码矩阵
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        positional_encoding = torch.zeros(max_seq_length, embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # 将位置编码与输入进行相加
        positional_encoding = self.positional_encoding[:, :seq_length, :]
        x = x + positional_encoding

        return self.dropout(x)
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    def forward(self, src):
        output = self.encoder(src)
        return output
class TS_Attention(nn.Module):
    def __init__(self,out_features):
        super(TS_Attention, self).__init__()
        self.attention = nn.Linear(out_features, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        atten=self.softmax(self.attention(data))
        data=data*atten
        return  data
class TSG(nn.Module):
    def __init__(self, edge_dim,in_features, hidden_features, out_features,dim_feedforward,num_heads, num_layers):
        super(TSG, self).__init__()
        self.gat = GEN(edge_dim,in_features, hidden_features, out_features, num_heads)
        self.pos_encoding = PositionalEncoding(out_features)
        self.compare = TransformerEncoder(out_features, num_heads, dim_feedforward, num_layers)
        self.atten=TS_Attention(out_features)
        self.fc2= nn.Linear(out_features, int(out_features/2), bias=True)
        self.fc3 = nn.Linear(int(out_features/2), 2, bias=True)
        self.embed1 = nn.Linear(2, out_features,bias=True)
        self.drop=nn.Dropout(0.1)
        self.relu=nn.ReLU()
    def forward(self, x_list,current_label):#,tgt_mask, memory_mask,tgt_padding_mask):
        x_c=[]
        for x in x_list:
            x_c.append(self.gat(x).unsqueeze(1))
        out=torch.cat(x_c,dim=1)
        current_label=self.embed1(current_label)
        out=out+current_label
        out = self.pos_encoding(out)
        out = out.permute(1, 0, 2)
        out = self.compare(out)
        out = out.permute(1, 0, 2)
        out=self.atten(out)
        out=torch.sum(out,dim=1)
        out = self.fc2(self.drop(self.relu(out)))
        out_pred=self.fc3(out)
        return out_pred

