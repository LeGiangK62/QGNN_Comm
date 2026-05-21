import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import torch.nn.functional as F
# from torch.nn import ModuleList, Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, LayerNorm, Dropout, LeakyReLU

from torch.nn import Sequential as Seq, Linear as Lin, Sigmoid


# =========================
# Utils
# =========================

def uniform_pi_init(tensor):
    return nn.init.uniform_(tensor, a=0.0, b=np.pi)

def small_normal_init(tensor):
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.1)

def input_process(tensor):
    return torch.tanh(tensor) * np.pi


class MLP(nn.Module):
    def __init__(self, dims, act='leaky_relu', norm=None, dropout=0.0):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            is_last = (i == len(dims) - 2)

            layers.append(nn.Linear(in_dim, out_dim))

            if not is_last:
                if norm == 'batch_norm':
                    layers.append(nn.BatchNorm1d(out_dim))

                if act == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                elif act == 'relu':
                    layers.append(nn.ReLU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =========================
# Quantum circuit (batched)
# =========================

def meassage_circuit(strong, inits, wires):
    """
    wires = [edge_wire, center_wire, neighbor_wire]
    """
    w0, _, w1 = wires
    num_ent_layer = strong.shape[0]

    for i in range(num_ent_layer):
        qml.RY(inits[i, 0], wires=w0)
        qml.RY(inits[i, 1], wires=w1)

        qml.CRX(strong[i, 0], wires=[w0, w1])
        qml.CRX(strong[i, 1], wires=[w1, w0])

        qml.RY(inits[i, 2], wires=w0)
        qml.RY(inits[i, 3], wires=w1)

        qml.CRX(strong[i, 2], wires=[w0, w1])
        qml.CRX(strong[i, 3], wires=[w1, w0])


def qgcn_enhance_layer(inputs, strong, inits, update):
    """
    Batched PQC for a star-graphlet.

    inputs:
      - [flat_dim]      for a single center
      - [B, flat_dim]   for a batch of centers

    Layout per sample (feat_dim=2):
      [edge_0 ... edge_{K-1} | center | neigh_0 ... neigh_{K-1}]
    """
    feat_dim = 2
    num_edges = update.shape[0]
    num_nodes = num_edges + 1
    rows_per_sample = 2 * num_edges + 1

    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)

    batch_size = inputs.shape[0]
    inputs = inputs.reshape(batch_size, rows_per_sample, feat_dim)

    edge_start = 0
    node_start = num_edges
    aux_start = num_edges + num_nodes
    center_wire = node_start

    # 1) Encode edges
    for i in range(num_edges):
        qml.RY(inputs[:, i, 0], wires=edge_start + i)
        qml.RZ(inputs[:, i, 1], wires=edge_start + i)

    # 2) Encode nodes (center + neighbors)
    for i in range(num_nodes):
        qml.RY(inputs[:, num_edges + i, 0], wires=node_start + i)
        qml.RZ(inputs[:, num_edges + i, 1], wires=node_start + i)

    # 3) Entanglement edge ↔ (center, neighbor)
    for i in range(num_edges):
        neighbor_w = node_start + i + 1
        edge_w = edge_start + i
        meassage_circuit(
            strong=strong, inits=inits,
            wires=[edge_w, center_wire, neighbor_w],
        )

    # 4) Aggregate at center via auxiliary qubit
    for i in range(num_edges):
        neighbor_w = node_start + i + 1
        u_wires = [center_wire, neighbor_w, aux_start]
        qml.StronglyEntanglingLayers(weights=update[i], wires=u_wires)

    # 5) Readout
    return [
        qml.expval(qml.PauliZ(center_wire)),
        qml.expval(qml.PauliX(center_wire)),
        qml.expval(qml.PauliZ(aux_start)),
        qml.expval(qml.PauliX(aux_start)),
    ]


# =========================
# Heterogeneous QGNN (batched)
# =========================


class QGNN_UL(nn.Module):
    def __init__(
        self, 
        q_dev, 
        w_shapes, 
        hidden_dim, 
        node_input_dim={'UE': 1, 'AP': 1}, 
        edge_input_dim={'UE': 2, 'AP': 2},
        graphlet_size=4, 
        hop_neighbor=1, 
        meta=['UE', 'AP']
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.pqc_dim = 2 # number of feat per pqc for each node
        self.chunk = 1
        self.final_dim = self.pqc_dim * self.chunk # 2
        self.pqc_out = 4 # probs?
        
        self.max_neighbors = graphlet_size - 1

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
                norm='batch_norm', 
                dropout=0.1
            )

            self.input_edge[node_type] = MLP(
                [self.edge_input_dim[node_type], self.hidden_dim, self.pqc_dim],
                act='leaky_relu',
                norm='batch_norm', 
                dropout=0.1
            )

            for i in range(self.hop_neighbor):
                qnode = qml.QNode(qgcn_enhance_layer, q_dev, interface="torch")
                self.qconvs[f"lay{i+1}_{node_type}"] = qml.qnn.TorchLayer(qnode, w_shapes, init_method=small_normal_init)

                self.upds[f"lay{i+1}_{node_type}"] = MLP(
                    [self.pqc_dim + self.pqc_out, self.hidden_dim*2, self.hidden_dim*2, self.pqc_dim],
                    act='leaky_relu',
                    norm=None,
                    dropout=0.1
                )

                self.norms[f"lay{i+1}_{node_type}"] = nn.BatchNorm1d(self.pqc_dim)

        self.final_layer = MLP(
            [self.final_dim, self.hidden_dim*2, self.hidden_dim*2, 1],
            act='leaky_relu',
            norm='batch_norm', 
            dropout=0.1
        )

    def sampling_neighbors(self, neighbor_ids, edge_ids):
        if neighbor_ids.numel() > self.max_neighbors:
            perm = torch.randperm(neighbor_ids.numel(), device=neighbor_ids.device)[:self.max_neighbors]
            neighbor_ids = neighbor_ids[perm]
            edge_ids = edge_ids[perm]
        return neighbor_ids, edge_ids

    def _build_q_inputs_batch(self, src_feat, dst_feat, edge_attr, edge_index):
        """
        Builds batched PQC inputs for Heterogeneous graphs separating src (neighbors) 
        and dst (centers) node features.
        """
        device = src_feat.device
        dtype = src_feat.dtype

        dst_indices = torch.unique(edge_index[1])

        batched_inputs = []
        centers = []

        for center in dst_indices:
            center = center.long()

            neighbor_mask = (edge_index[1] == center)
            neighbor_ids = edge_index[0][neighbor_mask].long()
            edge_ids = torch.nonzero(neighbor_mask, as_tuple=False).view(-1).long()

            if neighbor_ids.numel() == 0:
                continue

            neighbor_ids, edge_ids = self.sampling_neighbors(neighbor_ids, edge_ids)

            # 1. Edge features: [K, pqc_dim]
            e_feat = torch.zeros((self.max_neighbors, self.pqc_dim), device=device, dtype=dtype)
            e_feat[:edge_ids.numel()] = edge_attr[edge_ids]

            # 2. Center feature (from dst_feat): [1, final_dim]
            center_feat = dst_feat[center].unsqueeze(0)

            # 3. Neighbor features (from src_feat): [K, final_dim]
            n_feat = torch.zeros((self.max_neighbors, self.final_dim), device=device, dtype=dtype)
            n_feat[:neighbor_ids.numel()] = src_feat[neighbor_ids]

            # Concat as [K edge rows ; 1 center row ; K neighbor rows] 
            sample_inputs = torch.cat([e_feat, center_feat, n_feat], dim=0)
            batched_inputs.append(sample_inputs.flatten())
            centers.append(center)

        if len(batched_inputs) == 0:
            return None, None

        q_inputs_batch = torch.stack(batched_inputs, dim=0)  # [B, flat_dim]
        center_ids = torch.stack(centers, dim=0).long()      # [B]
        
        return q_inputs_batch, center_ids

    def forward(self, x_dict, edge_attr_dict, edge_index_dict, batch_dict=None):       
         
        # Preprocess node features
        for node_type, node_feat in x_dict.items():
            x_dict[node_type] = self.input_node[node_type](node_feat.float())
            x_dict[node_type] = input_process(x_dict[node_type])

        # Preprocess edge features
        for edge_type, edge_attr in edge_attr_dict.items():
            src_type, _, dst_type = edge_type
            edge_attr_dict[edge_type] = self.input_edge[dst_type](edge_attr.float())
            edge_attr_dict[edge_type] = input_process(edge_attr_dict[edge_type])

        # Message Passing Layers
        for i in range(self.hop_neighbor):
            
            # Temporary storage to prevent updating features prematurely within the same hop
            updates_dict = {ntype: torch.zeros_like(x_dict[ntype]) for ntype in x_dict}
            
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type

                q_layer = self.qconvs[f"lay{i+1}_{dst_type}"]
                upd_layer = self.upds[f"lay{i+1}_{dst_type}"]

                edge_attr = edge_attr_dict[edge_type]
                src_feat = x_dict[src_type]
                dst_feat = x_dict[dst_type]
                
                # Retrieve batched inputs
                q_inputs_batch, center_ids = self._build_q_inputs_batch(
                    src_feat=src_feat,
                    dst_feat=dst_feat,
                    edge_attr=edge_attr,
                    edge_index=edge_index
                )
                
                if q_inputs_batch is None:
                    continue
                
                # Pass through Parameterized Quantum Circuit (Batch Execution)
                # Output shape: [B, pqc_out]
                all_msgs = q_layer(q_inputs_batch)
                
                # Layer norm on batched outputs (Replaces `all_msg = F.layer_norm(...)` inside the loop)
                all_msgs = F.layer_norm(all_msgs, (self.pqc_out,))  
                
                # Prepare update MLPs
                center_feat_batch = dst_feat[center_ids]                 # [B, 2]
                upd_in = torch.cat([center_feat_batch, all_msgs], dim=1) # [B, 2 + pqc_out]
                updates = upd_layer(upd_in)                              # [B, 2]
                
                # Accumulate scattered updates for this destination type
                updates_dict[dst_type] = updates_dict[dst_type].index_add(0, center_ids, updates)

            # Apply norms and residual connections after iterating through all edge types for this hop
            for node_type in x_dict.keys():
                norm_layer = self.norms[f"lay{i+1}_{node_type}"]
                x_dict[node_type] = norm_layer(x_dict[node_type] + updates_dict[node_type])

        return torch.sigmoid(self.final_layer(x_dict['UE']))
    

class QGNN_DL(nn.Module):
    def __init__(
        self,
        q_dev,
        w_shapes,
        hidden_dim,
        node_input_dim=1,
        edge_input_dim=1,
        graphlet_size=4,
        hop_neighbor=1,
        num_classes=2,
        one_hot=0,
    ):
        super().__init__()
        self.q_dev = q_dev
        self.hidden_dim = hidden_dim
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.one_hot = one_hot

        self.pqc_dim = 2
        self.final_dim = 2
        self.pqc_out = 4
        self.max_neighbors = graphlet_size - 1

        if self.one_hot:
            self.node_input_dim = 1
            self.edge_input_dim = 1
        else:
            self.node_input_dim = node_input_dim
            self.edge_input_dim = edge_input_dim if edge_input_dim > 0 else 1

        self.input_node = MLP(
            [self.node_input_dim, hidden_dim, self.final_dim],
            act='leaky_relu',
            norm='batch_norm',
            dropout=0.1,
        )

        self.input_edge = MLP(
            [self.edge_input_dim, hidden_dim, self.pqc_dim],
            act='leaky_relu',
            norm='batch_norm',
            dropout=0.1,
        )

        self.qconvs = nn.ModuleDict()
        self.upds = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        for i in range(self.hop_neighbor):

            qnode = qml.QNode(qgcn_enhance_layer, q_dev, interface="torch")

            self.qconvs[f"lay{i+1}"] = qml.qnn.TorchLayer(
                qnode=qnode,
                weight_shapes=w_shapes,
                init_method=uniform_pi_init,
            )

            self.upds[f"lay{i+1}"] = MLP(
                [self.pqc_dim + self.pqc_out, hidden_dim, self.pqc_dim],
                act='leaky_relu',
                norm=None,
                dropout=0.1,
            )

            self.norms[f"lay{i+1}"] = nn.LayerNorm(self.pqc_dim)

        self.final_layer = MLP(
            [self.final_dim, self.hidden_dim, 
            #  self.hidden_dim, 1
             ],
            act='leaky_relu', 
            norm='batch_norm', 
            dropout=0.1
        ) 
        self.final_layer = nn.Sequential(*[self.final_layer, Seq(Lin(self.hidden_dim, 1), Sigmoid())])
        # Same as baseline
        # self.final_layer = MLP([self.final_dim, self.hidden_dim])
        # self.final_layer = nn.Sequential(*[self.final_layer, Seq(Lin(self.hidden_dim, 1), Sigmoid())])

    def sampling_neighbors(self, neighbor_ids, edge_ids):
        if neighbor_ids.numel() > self.max_neighbors:
            perm = torch.randperm(
                neighbor_ids.numel(), device=neighbor_ids.device
            )[:self.max_neighbors]
            neighbor_ids = neighbor_ids[perm]
            edge_ids = edge_ids[perm]
        return neighbor_ids, edge_ids

    def _build_q_inputs_batch(self, node_features, edge_features, edge_index):
        """
        Build fixed-size inputs cho PQC.
        Mỗi center -> 1 sample có cùng flat_dim.
        """
        device = node_features.device
        dtype = node_features.dtype

        dst_indices = torch.unique(edge_index[:, 1])

        batched_inputs = []
        centers = []

        for center in dst_indices:
            center = center.long()

            neighbor_mask = (edge_index[:, 1] == center)
            neighbor_ids = edge_index[:, 0][neighbor_mask].long()
            edge_ids = torch.nonzero(neighbor_mask, as_tuple=False).view(-1).long()

            if neighbor_ids.numel() == 0:
                continue

            neighbor_ids, edge_ids = self.sampling_neighbors(neighbor_ids, edge_ids)

            # edge features: [K, 2]
            e_feat = torch.zeros(
                (self.max_neighbors, self.pqc_dim),
                device=device,
                dtype=dtype,
            )
            e_feat[:edge_ids.numel()] = edge_features[edge_ids]

            # center feature: [1, 2]
            center_feat = node_features[center].unsqueeze(0)

            # neighbor features: [K, 2]
            n_feat = torch.zeros(
                (self.max_neighbors, self.final_dim),
                device=device,
                dtype=dtype,
            )
            n_feat[:neighbor_ids.numel()] = node_features[neighbor_ids]

            # [K edge rows ; 1 center row ; K neighbor rows]
            sample_inputs = torch.cat([e_feat, center_feat, n_feat], dim=0)
            batched_inputs.append(sample_inputs.flatten())
            centers.append(center)

        if len(batched_inputs) == 0:
            return None, None

        q_inputs_batch = torch.stack(batched_inputs, dim=0)  # [B, flat_dim]
        center_ids = torch.stack(centers, dim=0).long()      # [B]
        return q_inputs_batch, center_ids

    def forward(self, batch):

        node_feat, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        
        edge_index = edge_index.t().contiguous()
        device = node_feat.device

        if edge_attr is None:
            edge_attr = torch.ones(
                (edge_index.size(0), self.edge_input_dim),
                device=device,
                dtype=torch.float32,
            )
        elif edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        node_features = self.input_node(node_feat.float())
        edge_features = self.input_edge(edge_attr.float())

        node_features = input_process(node_features)
        edge_features = input_process(edge_features)

        for i in range(self.hop_neighbor):
            q_layer = self.qconvs[f"lay{i+1}"]
            upd_layer = self.upds[f"lay{i+1}"]
            norm_layer = self.norms[f"lay{i+1}"]

            updates_node = torch.zeros_like(node_features)

            q_inputs_batch, center_ids = self._build_q_inputs_batch(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
            )
            # print(q_inputs_batch.shape)

            if q_inputs_batch is None:
                node_features = norm_layer(updates_node) + node_features
                continue

            # output: [B, 3]
            all_msgs = q_layer(q_inputs_batch)

            center_feat_batch = node_features[center_ids]            # [B, 2]
            upd_in = torch.cat([center_feat_batch, all_msgs], dim=1) # [B, 5]
            updates = upd_layer(upd_in)                              # [B, 2]

            updates_node = updates_node.index_add(0, center_ids, updates)
            node_features = norm_layer(updates_node) + node_features

        output = self.final_layer(node_features)
        # output = torch.sigmoid(output)
        # output = F.softplus(output)
        # output = torch.exp(-output)
        
        return output * 6