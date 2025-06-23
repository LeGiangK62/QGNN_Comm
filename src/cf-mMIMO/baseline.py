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



class Cf_edge_layer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
            
        self.upd_mlp = MLP([in_channels, out_channels, out_channels])
        self.msg_mlp = MLP([in_channels * 2, out_channels, out_channels])

    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(tmp)
    
    def update(self, aggr_out):
        return self.upd_mlp(aggr_out)

class GNN_Cf(nn.Module):
    def __init__(self, node_input_dim=1, edge_input_dim=1, num_layers=1, hidden_channels=64):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim   
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.input_node = MLP([node_input_dim, hidden_channels])
        self.input_edge = MLP([edge_input_dim, hidden_channels])
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(Cf_edge_layer(in_channels=hidden_channels, 
                                            out_channels=hidden_channels)
                              )
        
        self.final_layer = MLP([hidden_channels, hidden_channels, 1])
    
    def forward(self, node_feat, edge_attr, edge_index, batch):
        node_feat = self.input_node(node_feat)
        edge_attr = self.input_edge(edge_attr)
        

        # edge_index = edge_index.t()  # Ensure edge_index is in the correct format
        for conv in self.convs:
            node_feat = F.relu(conv(node_feat, edge_attr, edge_index))
        
        # node_feat = self.final_layer(node_feat)
        node_feat = torch.sigmoid(self.final_layer(node_feat))
        return node_feat
    
    
class GIN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels if i==0 else hidden_channels,
                       hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        return self.classifier(x)
    


class GCN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch  = in_channels if i==0 else hidden_channels
            out_ch = out_channels  if i==num_layers-1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        # last layer, no activation
        return self.convs[-1](x, edge_index)


class GATN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, heads=8, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i==0 else hidden_channels * heads
            if i < num_layers-1:
                self.convs.append(
                    GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout)
                )
            else:
                # final layer: single head, no concat
                self.convs.append(
                    GATConv(in_ch, out_channels, heads=1, concat=False, dropout=dropout)
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        return self.convs[-1](x, edge_index)
    
    
# NOTE: Graph 

class GIN_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP([in_channels if i==0 else hidden_channels,
                       hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
class GCN_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
class GAT_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(GATConv(in_dim, hidden_channels, heads=heads))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)


class GraphSAGE_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_dim, hidden_channels))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
    
class Transformer_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(TransformerConv(in_dim, hidden_channels, heads=heads))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)