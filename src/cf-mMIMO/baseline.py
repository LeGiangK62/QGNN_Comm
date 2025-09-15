import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, TransformerConv, MLP, global_add_pool
from torch_geometric.nn import MessagePassing, HeteroConv

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])
class EdgeConv(MessagePassing):
    def __init__(self, input_dim, node_dim, hidden_channels, **kwargs):
        super(EdgeConv, self).__init__(aggr='mean')  # mean aggregation
        self.lin = MLP([input_dim, hidden_channels])
        self.res_lin = Lin(node_dim, hidden_channels)
        self.bn = BN(hidden_channels)

    def forward(self, x, edge_index, edge_attr):

        feat_src, feat_dst = x


        out = self.propagate(edge_index=edge_index, x=(feat_src, feat_dst), edge_attr=edge_attr)


        return self.bn(out + self.res_lin(feat_dst))

    def message(self, x_j, x_i, edge_attr):
        out = torch.cat([x_j, x_i, edge_attr], dim=1)
        return self.lin(out)

    def update(self, aggr_out):
        return aggr_out


class RGCN(nn.Module):
    def __init__(self, node_input_dim=1, edge_input_dim=1, num_layers=1, hidden_channels=64):
        super(RGCN, self).__init__()
        
        
        input_dim = node_input_dim*2  + edge_input_dim
        
        self.conv1 = HeteroConv({
            ('UE', 'com-by', 'AP'): EdgeConv(input_dim, node_input_dim, hidden_channels),
            ('AP', 'com', 'UE'): EdgeConv(input_dim, node_input_dim, hidden_channels)
        }, aggr='mean')
        
        input_dim = hidden_channels * 2  + edge_input_dim

        self.conv2 = HeteroConv({
            ('UE', 'com-by', 'AP'): EdgeConv(input_dim, hidden_channels, hidden_channels),
            ('AP', 'com', 'UE'): EdgeConv(input_dim, hidden_channels, hidden_channels)
        }, aggr='mean')

        self.conv3 = HeteroConv({
            ('UE', 'com-by', 'AP'): EdgeConv(input_dim, hidden_channels, hidden_channels),
            ('AP', 'com', 'UE'): EdgeConv(input_dim, hidden_channels, hidden_channels)
        }, aggr='mean')

        self.mlp = MLP([hidden_channels, hidden_channels])
        self.mlp = nn.Sequential(*[self.mlp, Seq(Lin(hidden_channels, 1), Sigmoid())])

    def forward(self,x_dict, edge_attr_dict, edge_index_dict, batch_dict=None):
        out = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        out = self.conv2(out, edge_index_dict, edge_attr_dict)
        out = self.conv3(out, edge_index_dict, edge_attr_dict)
        out = self.mlp(out['UE'])
        return out
