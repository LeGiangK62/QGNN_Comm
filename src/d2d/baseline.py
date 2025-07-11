import torch
import torch.nn as nn
from torch.nn import ModuleList, Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, TransformerConv, MLP, global_add_pool
from torch_geometric.nn import MessagePassing


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias = True), ReLU())#, BN(channels[i]))
        for i in range(1, len(channels))
    ])
    
class IGConv(MessagePassing):
    def __init__(self, node_input_dim, edge_input_dim, hidden_channels, **kwargs):
        super(IGConv, self).__init__(aggr='mean', **kwargs)

        self.mlp1 = MLP([node_input_dim + edge_input_dim, hidden_channels, hidden_channels])
        self.mlp2 = MLP([node_input_dim + hidden_channels, hidden_channels])
        self.mlp2 = Seq(self.mlp2, Seq(Lin(hidden_channels, hidden_channels, bias=True), Sigmoid()))
        self.bn = BN(hidden_channels)
        #self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        
    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        x = self.mlp2(tmp)
        return x
        # return torch.cat([x[:,:2], comb],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(x + out)
        return out

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)


class IGCNet(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_channels, num_layers):
        super(IGCNet, self).__init__()
        self.num_layers = num_layers
        
        self.convs = ModuleList()
        
        self.input = MLP([node_input_dim, hidden_channels, hidden_channels])

        for _ in range(self.num_layers):
            # mlp1 = MLP([node_input_dim + edge_input_dim, hidden_channels, hidden_channels*2])
            # mlp2 = MLP([node_input_dim + hidden_channels*2, hidden_channels])
            # mlp2 = Seq(mlp2, Seq(Lin(hidden_channels, hidden_channels, bias=True), Sigmoid()))
            self.convs.append(IGConv(hidden_channels, edge_input_dim, hidden_channels))
            
        self.final = MLP([hidden_channels, hidden_channels])
        self.final = nn.Sequential(*[self.final, Seq(Lin(hidden_channels, 1), Sigmoid())])

    def forward(self, x, edge_attr, edge_index, batch):
        x = self.input(x)
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.final(x)
        return x
            
        # x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        # x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        # #x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # #x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        # out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # return out