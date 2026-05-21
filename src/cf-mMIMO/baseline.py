import torch
import torch.nn as nn
# from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, TransformerConv, MLP, global_add_pool
from torch_geometric.nn import MessagePassing, HeteroConv
from torch.nn import ModuleList, Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, LayerNorm, Dropout, LeakyReLU
from torch_geometric.nn.inits import glorot, reset


# def MLP(channels, batch_norm=True):
#     return Seq(*[
#         Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
#         for i in range(1, len(channels))
#     ])


def MLP(channels, batch_norm=False, dropout_prob=0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Seq(Lin(channels[i - 1], channels[i])))
        if batch_norm:
            layers.append(LayerNorm(channels[i]))
        if dropout_prob:
            layers.append(Dropout(dropout_prob))  # Add dropout after batch norm or activation
        layers.append(LeakyReLU(negative_slope=0.1))
        # layers.append(GELU())
    return Seq(*layers)
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
    
# Homo graph neural network
class HomoCfmMIMOConv(MessagePassing):
    def __init__(self, node_hidden_dim, edge_input_dim, **kwargs):
        """
        Message Passing Layer cho Homogeneous Graph.
        - node_hidden_dim: Kích thước feature của node sau khi đã qua lớp nhúng (input MLP).
        - edge_input_dim: Kích thước feature của cạnh (thường là 4: nhiễu chéo 2 chiều beta/gamma).
        """
        super(HomoCfmMIMOConv, self).__init__(aggr='add', **kwargs)

        # Message MLP: Kết hợp feature node hàng xóm (x_j) và feature cạnh (edge_attr)
        # Input: node_hidden_dim + edge_input_dim -> Output: node_hidden_dim * 2
        self.msg = MLP([node_hidden_dim + edge_input_dim, node_hidden_dim, node_hidden_dim * 2])
        
        # Update MLP: Kết hợp feature node hiện tại (x) và thông tin tổng hợp từ hàng xóm (aggr_out)
        # Input: node_hidden_dim (từ x) + node_hidden_dim*2 (từ aggr_out) -> Output: node_hidden_dim
        self.upd = MLP([node_hidden_dim + node_hidden_dim * 2, node_hidden_dim])
        self.upd = Seq(self.upd, Seq(Lin(node_hidden_dim, node_hidden_dim, bias=True), Sigmoid()))
        
        self.bn = BN(node_hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        out = self.bn(x + out)
        return out

    def message(self, x_j, edge_attr):
        # x_j: src node - neighbor
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.msg(tmp)
        return agg

    def update(self, aggr_out, x):
        # x: dst node feature
        tmp = torch.cat([x, aggr_out], dim=1)
        out = self.upd(tmp)
        return out


class HomoCfmMimoNet(torch.nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_channels, num_layers):
        """
        - node_input_dim: 3 - beta, gamma, phi
        - edge_input_dim: 4 - h_src_dst, h_dst_src ?
        """
        super(HomoCfmMimoNet, self).__init__()
        self.num_layers = num_layers
        
        # Input Layer
        self.input_mlp = MLP([node_input_dim, hidden_channels, hidden_channels])

        # Graph Conv
        self.convs = ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(HomoCfmMIMOConv(
                node_hidden_dim=hidden_channels, 
                edge_input_dim=edge_input_dim
            ))
            
        # Power preidiction
        # self.final_mlp = MLP([hidden_channels, hidden_channels])
        # self.final = nn.Sequential(
        #     self.final_mlp,
        #     Lin(hidden_channels, 1),
        #     Sigmoid()
        # )

        self.final = MLP([hidden_channels, hidden_channels//2], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.final = nn.Sequential(
            *[
                self.final, Seq(Lin(hidden_channels//2, 1)), 
            ]
        )

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.input_mlp(x)
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = self.final(x)
        return out




    

# Heterogeneous GNN
class APConvLayer(MessagePassing):
    def __init__(
            self,
            src_dim_dict,
            edge_dim,
            out_channel,
            init_channel,
            metadata,
            drop_p=0,
            **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.metadata = metadata
        self.src_init_dict = init_channel
        self.edge_init = init_channel['edge']
        self.out_channel = out_channel
        self.src_dim_dict = src_dim_dict
        self.drop_prob = drop_p

        self.msg = nn.ModuleDict() 
        self.upd = nn.ModuleDict() 
        self.edge_upd = nn.ModuleDict() 
        
        self.gamma = nn.ParameterDict()
        self.gamma_edge = nn.ParameterDict()
        
        hidden = out_channel//2
        for edge_type in metadata:
            src_type, short_edge_type, dst_type = edge_type
            src_dim = src_dim_dict[src_type]
            dst_dim = src_dim_dict[dst_type]
            src_init = init_channel[src_type]
            dst_init = init_channel[dst_type]
            self.msg[src_type] = MLP(
                [src_dim + edge_dim, hidden], 
                batch_norm=False, dropout_prob=0.1
            )
            self.upd[dst_type] = MLP(
                [hidden + dst_dim, out_channel - dst_init], 
                batch_norm=False, dropout_prob=0.1
            )
            
            self.edge_upd[short_edge_type] = MLP(
                [sum(src_dim_dict.values()) + edge_dim, out_channel - self.edge_init], 
                batch_norm=False, dropout_prob=0.1
            )

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.msg)
        reset(self.upd)
        reset(self.edge_upd)

    def forward(
            self,
            x_dict,
            edge_index_dict,
            edge_attr_dict
    ):
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in self.metadata: continue;
            src_type, _, dst_type = edge_type

            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]
            
            edge_attr = edge_attr_dict[edge_type]
            

            # Node update                
            msg = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)
            tmp = torch.cat([x_dst, msg], dim=1)
            tmp = self.upd[dst_type](tmp)
            src_init_dim = self.src_init_dict[dst_type]
            if self.src_dim_dict[dst_type] == self.out_channel:
                tmp = tmp +  x_dst[:,src_init_dim:] # * self.gamma[dst_type]
            x_dict[dst_type] = torch.cat([x_dst[:,:src_init_dim], tmp], dim=1)
            # Edge update
            edge_attr_dict[edge_type] = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)

        return x_dict, edge_attr_dict

    def message(self, x_j, x_i, edge_attr, edge_type):
        # x_j: source node
        # x_i: destination node
        src_type, _, dst_type = edge_type
        out = torch.cat([x_j, edge_attr], dim=1)
        out = self.msg[src_type](out)
        return out

    def edge_update(self, x_j, x_i, edge_attr, edge_type):
        _, short_edge_type, _ = edge_type
        tmp = torch.cat([x_j, edge_attr, x_i], dim=1)
        out = self.edge_upd[short_edge_type](tmp)
        
        if self.out_channel == self.edge_init:
            out = out + edge_attr # * self.gamma_edge
        out = torch.cat([edge_attr[:,:self.edge_init], out], dim=1)
        return out



# Centralized GNN

class APHetNet(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNet, self).__init__()
        src_dim_dict = dim_dict.copy()

        self.ue_dim = src_dim_dict['UE']
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']
        
        self.convs = torch.nn.ModuleList()        
        # First Layer to update RRU
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': self.ap_dim},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('UE', 'up', 'AP')],
            )
        )
        
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': out_channels},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('AP', 'down', 'UE')],
            )
        )
    
        # Multiple conv layer for AP - UE
        for _ in range(num_layers):
            conv = APConvLayer(
                {'UE': out_channels, 'AP': out_channels}, 
                out_channels, out_channels, src_dim_dict, 
                [('UE', 'up', 'AP'), ('AP', 'down', 'UE')],
                # drop_p=0.2
            )
            self.convs.append(conv)


        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )
    
        
    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        
        edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
        edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict