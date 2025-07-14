import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, MLP
from torch.nn import BatchNorm1d as BN


from utils import star_subgraph


# def message_passing_pqc_aux(strong, twodesign, inits, wires):
#     edge, center, neighbor, ancilla1, ancilla2 = wires

#     qml.CRX(inits[0, 0], wires=[neighbor, ancilla1])
#     qml.CRY(inits[0, 1], wires=[edge, ancilla1])
#     qml.CRZ(inits[0, 2], wires=[neighbor, ancilla2])
#     qml.CRY(inits[0, 3], wires=[edge, ancilla2])
#     qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, neighbor, ancilla1])
#     qml.StronglyEntanglingLayers(weights=strong[1], wires=[ancilla1, neighbor, ancilla2])


# def qgcn_enhance_layer_aux(inputs, spreadlayer, strong, twodesign, inits, update):
#     edge_feat_dim = feat_dim = node_feat_dim = 2
#     inputs = inputs.reshape(-1,feat_dim)
    
#     # The number of avaible nodes and edges
#     total_shape = inputs.shape[0]
#     num_nodes = (total_shape+1)//2
#     num_edges = num_nodes - 1
    
#     adjacency_matrix, vertex_features = inputs[:num_edges,:], inputs[num_edges:,:]

#     # The number of qubits assiged to each node and edge
#     num_qbit = spreadlayer.shape[1]
#     num_nodes_qbit = (num_qbit+1)//2
#     num_edges_qbit = num_nodes_qbit - 1
    
#     center_wire = num_edges_qbit
    
    
#     for i in range(num_edges):
#         qml.RY(adjacency_matrix[i][0], wires=i)
#         qml.RZ(adjacency_matrix[i][1], wires=i)
#         # qml.RX(adjacency_matrix[i][2], wires=i)
    
#     for i in range(num_nodes):
#         qml.RY(vertex_features[i][0], wires=center_wire+i)
#         qml.RZ(vertex_features[i][1], wires=center_wire+i)
#         # qml.RX(vertex_features[i][2], wires=center_wire+i)
    
    
#     for i in range(num_edges):

#         message_passing_pqc_aux(strong=strong, twodesign=twodesign, inits=inits, 
#                             wires=[i, center_wire, center_wire+i+1, num_qbit, num_qbit+1])

#     qml.StronglyEntanglingLayers(
#         weights=update[0], 
#         wires=[center_wire, num_qbit, num_qbit+1]
#         )
#     # probs = qml.probs(wires=[center_wire, num_qbit, num_qbit+1])
#     # return probs
#     # expval = [qml.expval(qml.PauliZ(w)) for w in [center_wire, num_qbit, num_qbit+1]]
#     expval = [
#         qml.expval(qml.PauliZ(center_wire)),
#         qml.expval(qml.PauliZ(num_qbit)),
#         qml.expval(qml.PauliZ(num_qbit+1))
#     ]
#     # expval = [
#     #     qml.expval(qml.PauliZ(center_wire)),
#     #     qml.expval(qml.PauliX(center_wire)),
#     #     qml.expval(qml.PauliZ(num_qbit)),
#     #     qml.expval(qml.PauliX(num_qbit)),
#     #     qml.expval(qml.PauliZ(num_qbit+1)),
#     #     qml.expval(qml.PauliX(num_qbit+1))
#     # ]
#     return expval

## Todo: New Approach, Message and Aggregate Seperate
def message_passing_pqc(strong, twodesign, inits, wires):
    edge, center, neighbor = wires

    qml.CRX(inits[0, 0], wires=[neighbor, edge])
    qml.CRY(inits[0, 1], wires=[edge, neighbor])
    # qml.CRZ(inits[0, 2], wires=[neighbor, ancilla2])
    # qml.CRY(inits[0, 3], wires=[edge, ancilla2])
    qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, neighbor])
    # qml.StronglyEntanglingLayers(weights=strong[1], wires=[ancilla1, neighbor, ancilla2])
    

def qgcn_enhance_layer(inputs, spreadlayer, strong, twodesign, inits, update):
    edge_feat_dim = feat_dim = node_feat_dim = 2
    inputs = inputs.reshape(-1,feat_dim)
    
    # The number of avaible nodes and edges
    total_shape = inputs.shape[0]
    num_nodes = (total_shape+1)//2
    num_edges = num_nodes - 1
    
    adjacency_matrix, vertex_features = inputs[:num_edges,:], inputs[num_edges:,:]

    # The number of qubits assiged to each node and edge
    num_qbit = spreadlayer.shape[1]
    num_nodes_qbit = (num_qbit+1)//2
    num_edges_qbit = num_nodes_qbit - 1
    
    center_wire = num_edges_qbit
    
    
    for i in range(num_edges):
        qml.RX(adjacency_matrix[i][0], wires=i)
        qml.RZ(adjacency_matrix[i][1], wires=i)
        # qml.RX(adjacency_matrix[i][2], wires=i)
    
    for i in range(num_nodes):
        qml.RX(vertex_features[i][0], wires=center_wire+i)
        qml.RZ(vertex_features[i][1], wires=center_wire+i)
        # qml.RX(vertex_features[i][2], wires=center_wire+i)
    
    
    for i in range(num_edges):
        message_passing_pqc(strong=strong, twodesign=twodesign, inits=inits, 
                            wires=[i, center_wire, center_wire+i+1])
    
    for i in range(num_edges):
        # # No auxiliary
        # qml.StronglyEntanglingLayers(weights=update[i],wires=[center_wire, center_wire+i+1])
        # 2 qubit auxiliary
        qml.StronglyEntanglingLayers(weights=update[i],wires=[center_wire, center_wire+i+1, num_qbit])

    # probs = qml.probs(wires=[center_wire, num_qbit, num_qbit+1])
    # return probs
    # expval = [qml.expval(qml.PauliZ(w)) for w in [center_wire, num_qbit, num_qbit+1]]
    expval = [
        qml.expval(qml.PauliZ(center_wire)),
        qml.expval(qml.PauliZ(num_qbit)),
    ]
    # expval = [
    #     qml.probs(wires=[center_wire])
    # ]
    return expval


def small_normal_init(tensor):
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.1)

def uniform_pi_init(tensor):
    return nn.init.uniform_(tensor, a=0.0, b=np.pi)

def identity_block_init(tensor):
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndim < 1:
            return tensor  # scalar param

        # Total number of parameters
        total_params = tensor.numel()
        num_active = max(1, total_params // 3)

        # Flatten, randomize, and reshape
        flat = tensor.view(-1)
        active_idx = torch.randperm(flat.shape[0])[:num_active]
        flat[active_idx] = torch.randn_like(flat[active_idx]) * 0.1

        return tensor
    
def input_process(tensor):
    # return torch.clamp(tensor, -1.0, 1.0) * np.pi
    return torch.tanh(tensor) * np.pi

class QGNN(nn.Module):
    def __init__(self, q_dev, w_shapes, hidden_dim, node_input_dim={'UE': 1, 'AP': 1}, edge_input_dim={'UE': 2, 'AP': 2},
                 graphlet_size=4, hop_neighbor=1, meta=['UE', 'AP']):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.pqc_dim = 2 # number of feat per pqc for each node
        self.chunk = 1
        self.final_dim = self.pqc_dim * self.chunk # 2
        self.pqc_out = 2 # probs?


        self.input_node = nn.ModuleDict()
        self.input_edge = nn.ModuleDict()
        self.qconvs = nn.ModuleDict()
        self.upds = nn.ModuleDict()
        self.aggs = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        
        self.node_input_dim = {}
        self.edge_input_dim = {}

        print(f"Hidden dim: {self.hidden_dim}")

        for node_type in meta:
            self.node_input_dim[node_type] = node_input_dim[node_type]
            self.edge_input_dim[node_type] = edge_input_dim[node_type] if edge_input_dim[node_type] > 0 else 1
            self.input_node[node_type] = MLP(
                [self.node_input_dim[node_type], self.hidden_dim, self.final_dim],
                act='leaky_relu',
                # norm=None,
                norm='batch_norm', 
                dropout=0.1
            )

            self.input_edge[node_type] = MLP(
                [self.edge_input_dim[node_type], self.hidden_dim, self.pqc_dim],
                act='leaky_relu',
                # norm=None,
                norm='batch_norm', 
                dropout=0.1
            )

            for i in range(self.hop_neighbor):
                qnode = qml.QNode(qgcn_enhance_layer, q_dev,  interface="torch")
                self.qconvs[f"lay{i+1}_{node_type}"] = qml.qnn.TorchLayer(qnode, w_shapes, uniform_pi_init)

                self.upds[f"lay{i+1}_{node_type}"] = MLP(
                        [self.pqc_dim + self.pqc_out, self.hidden_dim, self.hidden_dim, self.pqc_dim],
                        act='leaky_relu',
                        norm=None,
                        dropout=0.1
                )

                # self.norms[f"lay{i+1}_{node_type}"] = nn.LayerNorm(self.pqc_dim)
                self.norms[f"lay{i+1}_{node_type}"] = nn.BatchNorm1d(self.pqc_dim)

        self.final_layer = MLP(
                [self.final_dim, self.hidden_dim, self.hidden_dim, 1],
                act='leaky_relu',
                # norm=None,
                norm='batch_norm', 
                dropout=0.1
        )

    def sampling_neighbors(self, neighbor_ids, edge_ids):
        # if neighbor_ids.numel() == 0:
        #     return neighbor_ids, edge_ids

        if neighbor_ids.numel() > self.graphlet_size - 1:
            perm = torch.randperm(neighbor_ids.numel())[:self.graphlet_size - 1]
            neighbor_ids = neighbor_ids[perm]
            edge_ids = edge_ids[perm]
            
        return neighbor_ids, edge_ids

    def forward(self, x_dict, edge_attr_dict, edge_index_dict, batch_dict):        
        # Prepocess node and edge features - To the same feature dim
        for node_type, node_feat in x_dict.items():
            # x_dict[node_type] = input_process(self.input_node[node_type](node_feat.float()))
            x_dict[node_type] = self.input_node[node_type](node_feat.float())

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            # edge_attr_dict[edge_type] = input_process(self.input_edge[dst_type](edge_attr_dict[edge_type].float()))
            edge_attr_dict[edge_type] = self.input_edge[dst_type](edge_attr_dict[edge_type].float())

        for i in range(self.hop_neighbor):
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type

                q_layer = self.qconvs[f"lay{i+1}_{dst_type}"]
                upd_layer = self.upds[f"lay{i+1}_{dst_type}"]
                norm_layer = self.norms[f"lay{i+1}_{dst_type}"]

                edge_attr = edge_attr_dict[edge_type]
                src_feat = x_dict[src_type]
                dst_feat = x_dict[dst_type]
                
                dst_indices = torch.unique(edge_index[1]).tolist()
                
                
                updates = []    
                centers = []
                    
                for dst_idx in dst_indices:
                    neighbor_mask = (edge_index[1] == dst_idx)
                    neighbor_ids = edge_index[0][neighbor_mask]
                    edge_ids = torch.nonzero(neighbor_mask, as_tuple=False).squeeze()
                    neighbor_ids, edge_ids  = self.sampling_neighbors(neighbor_ids, edge_ids)
                    
                    center = dst_feat[dst_idx]
                    neighbors = src_feat[neighbor_ids]
                    n_feat = torch.cat([center.unsqueeze(0), neighbors], dim=0)
                    e_feat = edge_attr[edge_ids.view(-1)]
                    
                    inputs = torch.cat([e_feat, n_feat], dim=0)
                    all_msg = q_layer(inputs.flatten())
                    aggr = all_msg
                    update_dst = upd_layer(torch.cat([center, aggr], dim=0))
                    updates.append(update_dst)
                    centers.append(dst_idx)
                    
                
                centers = torch.tensor(centers, device=dst_feat.device)
                updates = torch.stack(updates, dim=0)
                updates_node = torch.zeros_like(dst_feat)
                updates_node = updates_node.index_add(0, centers, updates)

                # node_features = norm_layer(updates_node + node_features)
                x_dict[dst_type] = norm_layer(x_dict[dst_type] + updates_node)
                # x_dict[dst_type] = input_process(x_dict[dst_type]) # do not use

        # return torch.sigmoid(self.final_layer(x_dict['UE']))
        return torch.exp(-F.softplus(self.final_layer(x_dict['UE'])))
    