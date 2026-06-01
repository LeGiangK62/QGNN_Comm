# VERSION: QAttention Context - neighbor_feat + edge_feat, random train / top-k eval
# CHECK: grep -n "qattention_ladder_circuit\|neighbor_feat" main_sumrate_DL_QAttention_ladder_v2.py
'''
Train homogeneous QGNN for downlink sum-rate cf-mMIMO with Quantum Attention top-k sampling.

This file is intentionally self-contained so it can be run like main_sumrate_DL.py.

Main changes compared with main_sumrate_DL.py:
  1. Homogeneous graph still has AP-UE link nodes.
  2. Edge attributes include two relation flags: same_ap and same_ue.
  3. Stargraph candidates are filtered to same-AP / same-UE neighbors.
  4. A small QNN attention scorer ranks candidate edges for each center node.
  5. Hard top-k selected neighbors are passed to the main QGNN graphlet PQC.

Run quick test:

python main_sumrate_DL_QAttention.py \
  --num_train 20 --num_test 10 --num_eval 10 \
  --batch_size 2 --graphlet_size 3 \
  --num_epochs 2 --num_epochs_cen 2

Only train QAttention-QGNN:

python main_sumrate_DL_QAttention.py \
  --num_train 20 --num_test 10 --num_eval 10 \
  --batch_size 2 --graphlet_size 3 \
  --num_epochs 2 --skip_cen
'''

import os
import time
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import scipy.io
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, Sigmoid
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cf_utils import (
    train_sumrate_homo,
    eval_sumrate_homo,
    loss_function_sumrate_homo,
)
from baseline import HomoCfmMimoNet
from cf_model import MLP, input_process, uniform_pi_init, qgcn_enhance_layer


# =====================================================================
# Paths
# =====================================================================

root_dir = '../..'
SAVE_DIR = os.path.join(root_dir, 'results', 'cf_sumrate')
MODEL_DIR = os.path.join(SAVE_DIR, 'models')
EVAL_DIR = os.path.join(SAVE_DIR, 'eval')
FIG_DIR = os.path.join(SAVE_DIR, 'figs')
TRAIN_DIR = os.path.join(SAVE_DIR, 'train')


def init_folder():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)


# =====================================================================
# Homogeneous graph construction with relation flags
# =====================================================================

def full_homo_graph_qattention(
    beta_single_sample,
    gamma_single_sample,
    label_single_all,
    phi_single_sample,
    ap_id=None,
    sample_id=None,
    device=None,
):
    """
    Build homogeneous AP-UE link graph.

    Node i represents one AP-UE link (m, k):
        i = m * num_UE + k

    Node feature:
        [beta_mk, gamma_mk, phi_k]

    Edge feature:
        [
            beta_srcAP_to_dstUE,
            gamma_srcAP_to_dstUE,
            beta_dstAP_to_srcUE,
            gamma_dstAP_to_srcUE,
            same_ap,
            same_ue,
        ]

    same_ap/same_ue are only used for candidate filtering before QAttention.
    The model itself does not need num_ue or num_ap.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_AP, num_UE = beta_single_sample.shape
    num_nodes = num_AP * num_UE

    ap_indices = np.repeat(np.arange(num_AP), num_UE)
    ue_indices = np.tile(np.arange(num_UE), num_AP)

    # ---------------------- Node features ----------------------
    node_beta = beta_single_sample.flatten()
    node_gamma = gamma_single_sample.flatten()

    if phi_single_sample.ndim == 1:
        node_phi = phi_single_sample[ue_indices].reshape(-1, 1)
    else:
        node_phi = phi_single_sample[ue_indices]

    x_features = np.column_stack((node_beta, node_gamma, node_phi))
    x = torch.tensor(x_features, dtype=torch.float32, device=device)

    # ---------------------- Fully connected directed edges ----------------------
    # We keep the full homograph. The QGNN stargraph sampler will filter
    # candidates to same-AP / same-UE before attention top-k.
    src, dst = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))
    src = src.flatten()
    dst = dst.flatten()

    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    edge_index = torch.tensor(
        np.vstack((src, dst)),
        dtype=torch.long,
        device=device,
    ).contiguous()

    ap_src = src // num_UE
    ue_src = src % num_UE
    ap_dst = dst // num_UE
    ue_dst = dst % num_UE

    beta_src_to_dst = beta_single_sample[ap_src, ue_dst]
    gamma_src_to_dst = gamma_single_sample[ap_src, ue_dst]

    beta_dst_to_src = beta_single_sample[ap_dst, ue_src]
    gamma_dst_to_src = gamma_single_sample[ap_dst, ue_src]

    same_ap = (ap_src == ap_dst).astype(np.float32)
    same_ue = (ue_src == ue_dst).astype(np.float32)

    edge_features = np.column_stack((
        beta_src_to_dst,
        gamma_src_to_dst,
        beta_dst_to_src,
        gamma_dst_to_src,
        same_ap,
        same_ue,
    ))

    edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=None)
    data.ap_id = ap_id
    data.sample_id = sample_id
    data.num_AP = num_AP
    data.num_UE = num_UE

    return data


def create_homo_graph_qattention(Beta_all, Gamma_all, Phi_all, device):
    num_sample, _, _ = Beta_all.shape
    data_list = []

    for each_sample in range(num_sample):
        data = full_homo_graph_qattention(
            Beta_all[each_sample],
            Gamma_all[each_sample],
            None,
            Phi_all[each_sample],
            sample_id=each_sample,
            device=device,
        )
        data_list.append(data)

    return data_list


def build_homo_loader_qattention(
    betaMatrix,
    gammaMatrix,
    phiMatrix,
    batchSize,
    isShuffle=False,
    device=None,
):
    log_large_scale = np.log1p(betaMatrix)
    data_cen = create_homo_graph_qattention(
        log_large_scale,
        gammaMatrix,
        phiMatrix,
        device=device,
    )
    loader_cen = DataLoader(data_cen, batch_size=batchSize, shuffle=isShuffle)
    return data_cen, loader_cen


# =====================================================================
# QAttention circuit
# =====================================================================

def small_normal_init(tensor):
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.1)

def qattention_ladder_circuit(inputs, weights):
    """
    QNN attention scorer.

    Input layout per candidate:
        [neighbor_feat(2), edge_feat(2)] = 4 dims

    wires:
        wire 0: neighbor/readout qubit
        wire 1: middle interaction qubit
        wire 2: edge/control qubit

    Ladder:
        wire 2 -> wire 1 -> wire 0
    """
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)

    neighbor_feat = inputs[:, 0:2]
    edge_feat = inputs[:, 2:4]

    # Encode neighbor information.
    qml.RY(neighbor_feat[:, 0], wires=0)
    qml.RZ(neighbor_feat[:, 1], wires=0)

    # Encode mixed neighbor-edge information.
    qml.RY(neighbor_feat[:, 0] + edge_feat[:, 0], wires=1)
    qml.RZ(neighbor_feat[:, 1] + edge_feat[:, 1], wires=1)

    # Encode edge information.
    qml.RY(edge_feat[:, 0], wires=2)
    qml.RZ(edge_feat[:, 1], wires=2)

    for l in range(weights.shape[0]):
        qml.RY(weights[l, 0], wires=0)
        qml.RZ(weights[l, 1], wires=0)

        qml.RY(weights[l, 2], wires=1)
        qml.RZ(weights[l, 3], wires=1)

        qml.RY(weights[l, 4], wires=2)
        qml.RZ(weights[l, 5], wires=2)

        # Ladder control from edge -> middle -> neighbor/readout.
        qml.CRX(weights[l, 6], wires=[2, 1])
        qml.CRZ(weights[l, 7], wires=[2, 1])

        qml.CRX(weights[l, 8], wires=[1, 0])
        qml.CRZ(weights[l, 9], wires=[1, 0])

    return qml.expval(qml.PauliZ(0))

class QAttentionScorer(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()

        self.q_dev = qml.device("default.qubit", wires=3)

        qnode = qml.QNode(
            qattention_ladder_circuit,
            self.q_dev,
            interface="torch",
        )

        self.q_layer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"weights": (num_layers, 10)},
            init_method=small_normal_init,
        )

    def forward(self, node_feat, edge_feat):
        """
        node_feat: [N, 2] candidate neighbor feature
        edge_feat: [N, 2] edge feature from neighbor to center
        """
        attn_input = torch.cat([node_feat, edge_feat], dim=1)  # [N, 4]
        raw_score = self.q_layer(attn_input)

        if raw_score.ndim > 1:
            raw_score = raw_score.view(-1)

        return torch.sigmoid(raw_score)


class QGNN_DL_QAttention_Context(nn.Module):
    """
    Improved QGNN for DL sum-rate.

    Differences from previous QAttention model:
        1. Training uses random sampling like UL.
        2. Evaluation uses QAttention top-k.
        3. QAttention still gates selected edge features.
        4. Adds classical all-candidate context aggregation.
        5. Uses residual normalization as norm(x + update).
        6. Final head returns raw logits, no sigmoid, no *6.
    """

    def __init__(
        self,
        q_dev,
        w_shapes,
        hidden_dim,
        node_input_dim=1,
        edge_input_dim=1,
        graphlet_size=4,
        hop_neighbor=1,
        one_hot=0,
        relation_filter=True,
        qattn_layers=1,
        context_aggr="mean",
    ):
        super().__init__()

        self.q_dev = q_dev
        self.hidden_dim = hidden_dim
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.one_hot = one_hot
        self.relation_filter = relation_filter
        self.context_aggr = context_aggr

        self.pqc_dim = 2
        self.final_dim = 2
        self.pqc_out = 4
        self.context_dim = 2
        self.max_neighbors = graphlet_size - 1

        if self.one_hot:
            self.node_input_dim = 1
            self.edge_input_dim = 1
        else:
            self.node_input_dim = node_input_dim
            self.edge_input_dim = edge_input_dim if edge_input_dim > 0 else 1

        self.input_node = MLP(
            [self.node_input_dim, hidden_dim, self.final_dim],
            act="leaky_relu",
            norm="batch_norm",
            dropout=0.1,
        )

        self.input_edge = MLP(
            [self.edge_input_dim, hidden_dim, self.pqc_dim],
            act="leaky_relu",
            norm="batch_norm",
            dropout=0.1,
        )

        self.attn = QAttentionScorer(num_layers=qattn_layers)

        self.qconvs = nn.ModuleDict()
        self.upds = nn.ModuleDict()
        self.norms = nn.ModuleDict()

        # Update input now includes:
        #   center feature      : pqc_dim = 2
        #   main PQC message    : pqc_out = 4
        #   classical context   : context_dim = 2
        update_in_dim = self.pqc_dim + self.pqc_out + self.context_dim

        for i in range(self.hop_neighbor):
            qnode = qml.QNode(qgcn_enhance_layer, q_dev, interface="torch")

            self.qconvs[f"lay{i+1}"] = qml.qnn.TorchLayer(
                qnode=qnode,
                weight_shapes=w_shapes,
                init_method=uniform_pi_init,
            )

            self.upds[f"lay{i+1}"] = MLP(
                [update_in_dim, hidden_dim * 2, hidden_dim, self.pqc_dim],
                act="leaky_relu",
                norm=None,
                dropout=0.1,
            )

            # LayerNorm is safer for tiny pqc_dim=2 than BatchNorm.
            self.norms[f"lay{i+1}"] = nn.LayerNorm(self.pqc_dim)

        self.final_layer = MLP(
            [self.final_dim, hidden_dim, hidden_dim // 2],
            act="leaky_relu",
            norm="batch_norm",
            dropout=0.1,
        )
        self.final_layer = nn.Sequential(
            self.final_layer,
            Lin(hidden_dim // 2, 1),
        )

    def filter_relation_neighbors(self, neighbor_ids, edge_ids, raw_edge_attr):
        """
        Keep only same-AP / same-UE candidates if flags are available.

        raw_edge_attr layout:
            [..., same_ap, same_ue]
        """
        if not self.relation_filter:
            return neighbor_ids, edge_ids

        if raw_edge_attr is None:
            return neighbor_ids, edge_ids

        if raw_edge_attr.ndim != 2 or raw_edge_attr.size(1) < 6:
            return neighbor_ids, edge_ids

        same_ap = raw_edge_attr[edge_ids, -2] > 0.5
        same_ue = raw_edge_attr[edge_ids, -1] > 0.5
        valid = same_ap | same_ue

        if valid.sum() == 0:
            return neighbor_ids, edge_ids

        return neighbor_ids[valid], edge_ids[valid]

    def compute_candidate_context(self, neighbor_ids, edge_ids, node_features, edge_features):
        """
        Classical context from all same-AP/same-UE candidates.

        This partially restores the "aggregate all neighbors" strength
        of the classical GNN baseline.
        """
        if neighbor_ids.numel() == 0:
            return torch.zeros(
                self.context_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )

        neigh_feat = node_features[neighbor_ids]      # [C, 2]
        cand_edge_feat = edge_features[edge_ids]      # [C, 2]

        # Simple message proxy: neighbor + edge.
        cand_msg = neigh_feat + cand_edge_feat        # [C, 2]

        if self.context_aggr == "sum":
            context = cand_msg.sum(dim=0)
        else:
            context = cand_msg.mean(dim=0)

        return context

    def select_neighbors_with_attention(
        self,
        neighbor_ids,
        edge_ids,
        node_features,
        edge_features,
    ):
        """
        Train:
            random sample like UL, then QAttention gates selected edges.

        Eval:
            deterministic top-k by QAttention score.
        """
        num_candidates = neighbor_ids.numel()

        if num_candidates == 0:
            return neighbor_ids, edge_ids, None

        total_k = min(self.max_neighbors, num_candidates)

        cand_neighbor_feat = node_features[neighbor_ids]
        cand_edge_feat = edge_features[edge_ids]

        score_gate = self.attn(
            node_feat=cand_neighbor_feat,
            edge_feat=cand_edge_feat,
        )

        if self.training:
            selected_idx = torch.randperm(
                num_candidates,
                device=neighbor_ids.device,
            )[:total_k]
        else:
            selected_idx = torch.topk(
                score_gate,
                k=total_k,
                largest=True,
                sorted=False,
            ).indices

        selected_neighbor_ids = neighbor_ids[selected_idx]
        selected_edge_ids = edge_ids[selected_idx]
        selected_score_gate = score_gate[selected_idx]

        return selected_neighbor_ids, selected_edge_ids, selected_score_gate

    def _build_q_inputs_batch(
        self,
        node_features,
        edge_features,
        edge_index,
        raw_edge_attr=None,
    ):
        """
        Returns:
            q_inputs_batch : [B, flat_dim]
            center_ids     : [B]
            context_batch  : [B, context_dim]
        """
        device = node_features.device
        dtype = node_features.dtype

        dst_indices = torch.unique(edge_index[:, 1])

        batched_inputs = []
        centers = []
        contexts = []

        for center in dst_indices:
            center = center.long()

            neighbor_mask = edge_index[:, 1] == center
            neighbor_ids = edge_index[:, 0][neighbor_mask].long()
            edge_ids = torch.nonzero(
                neighbor_mask,
                as_tuple=False,
            ).view(-1).long()

            if neighbor_ids.numel() == 0:
                continue

            # Restrict to same-AP / same-UE.
            neighbor_ids, edge_ids = self.filter_relation_neighbors(
                neighbor_ids=neighbor_ids,
                edge_ids=edge_ids,
                raw_edge_attr=raw_edge_attr,
            )

            if neighbor_ids.numel() == 0:
                continue

            # Classical context uses ALL filtered candidates.
            context = self.compute_candidate_context(
                neighbor_ids=neighbor_ids,
                edge_ids=edge_ids,
                node_features=node_features,
                edge_features=edge_features,
            )

            # PQC graphlet uses sampled/top-k subset.
            selected_neighbor_ids, selected_edge_ids, score_gate = self.select_neighbors_with_attention(
                neighbor_ids=neighbor_ids,
                edge_ids=edge_ids,
                node_features=node_features,
                edge_features=edge_features,
            )

            if selected_neighbor_ids.numel() == 0:
                continue

            e_feat = torch.zeros(
                (self.max_neighbors, self.pqc_dim),
                device=device,
                dtype=dtype,
            )

            selected_edge_feat = edge_features[selected_edge_ids]

            # QAttention remains trainable through this gate.
            if score_gate is not None:
                selected_edge_feat = selected_edge_feat * score_gate.unsqueeze(-1)

            e_feat[:selected_edge_ids.numel()] = selected_edge_feat

            center_feat = node_features[center].unsqueeze(0)

            n_feat = torch.zeros(
                (self.max_neighbors, self.final_dim),
                device=device,
                dtype=dtype,
            )
            n_feat[:selected_neighbor_ids.numel()] = node_features[selected_neighbor_ids]

            sample_inputs = torch.cat(
                [e_feat, center_feat, n_feat],
                dim=0,
            )

            batched_inputs.append(sample_inputs.flatten())
            centers.append(center)
            contexts.append(context)

        if len(batched_inputs) == 0:
            return None, None, None

        q_inputs_batch = torch.stack(batched_inputs, dim=0)
        center_ids = torch.stack(centers, dim=0).long()
        context_batch = torch.stack(contexts, dim=0)

        return q_inputs_batch, center_ids, context_batch

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

        raw_edge_attr = edge_attr.float()

        node_features = self.input_node(node_feat.float())
        edge_features = self.input_edge(edge_attr.float())

        # Keep the same angle encoding as current QGNN.
        node_features = input_process(node_features)
        edge_features = input_process(edge_features)

        for i in range(self.hop_neighbor):
            q_layer = self.qconvs[f"lay{i+1}"]
            upd_layer = self.upds[f"lay{i+1}"]
            norm_layer = self.norms[f"lay{i+1}"]

            updates_node = torch.zeros_like(node_features)

            q_inputs_batch, center_ids, context_batch = self._build_q_inputs_batch(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
                raw_edge_attr=raw_edge_attr,
            )

            if q_inputs_batch is None:
                node_features = norm_layer(node_features + updates_node)
                continue

            all_msgs = q_layer(q_inputs_batch)  # [B, pqc_out]

            center_feat_batch = node_features[center_ids]

            upd_in = torch.cat(
                [center_feat_batch, all_msgs, context_batch],
                dim=1,
            )

            updates = upd_layer(upd_in)

            updates_node = updates_node.index_add(
                0,
                center_ids,
                updates,
            )

            # Residual update without normalization.
            node_features = node_features + updates_node

        output = self.final_layer(node_features)

        # Raw logits for power_from_raw.
        return output


# =====================================================================
# Args
# =====================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description='Train QAttention-QGNN on cf-mMIMO downlink sum-rate maximization'
    )

    parser.add_argument('--seed', type=int, default=1712)
    parser.add_argument('--qgnn_pretrain', type=str, default=None)
    parser.add_argument('--c_pretrain', type=str, default=None)
    parser.add_argument('--eval_plot', action='store_true', default=True)
    parser.add_argument('--skip_cen', action='store_true', help='Skip centralized GNN benchmark')

    # System parameters
    parser.add_argument('--num_ap', type=int, default=30)
    parser.add_argument('--num_ue', type=int, default=6)
    parser.add_argument('--tau', type=int, default=20)
    parser.add_argument('--power_f', type=float, default=0.2)
    parser.add_argument('--D', type=float, default=1)
    parser.add_argument('--num_antenna', type=int, default=1)

    # Data
    parser.add_argument('--num_train', type=int, default=500)
    parser.add_argument('--num_test', type=int, default=200)
    parser.add_argument('--num_eval', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)

    # QGNN hyperparameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--lr_c', type=float, default=5e-3)
    parser.add_argument('--lr_q', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--graphlet_size', type=int, default=10)
    parser.add_argument('--node_qubit', type=int, default=3)
    parser.add_argument('--num_ent_layers', type=int, default=1)
    parser.add_argument('--qattn_layers', type=int, default=1)
    parser.add_argument('--q_dev', type=str, default='default.qubit')

    # Centralized benchmark hyperparameters
    parser.add_argument('--cen_lr', type=float, default=5e-3)
    parser.add_argument('--num_epochs_cen', type=int, default=50)
    parser.add_argument('--cen_hidden_channels', type=int, default=32)
    parser.add_argument('--cen_num_gnn_layers', type=int, default=3)
    parser.add_argument('--context_aggr', type=str, default='mean', choices=['mean', 'sum'])
    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    # Keep CPU default because default.qubit + many graphlets is often CPU-bound.
    device = torch.device('cpu')
    print(f'Using device: {device}')

    timestamp = time.strftime('%y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    init_folder()

    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    qgnn_pretrain = args.qgnn_pretrain
    c_pretrain = args.c_pretrain

    tau = args.tau
    num_antenna = args.num_antenna
    rho_p, rho_d = args.power_f, args.power_f
    num_ap = args.num_ap
    num_ue = args.num_ue

    # ---------------------- Load data ----------------------
    file_name = f'dl_sumrate_data_2000_{num_ue}_{num_ap}'
    data_path_candidates = [
        os.path.join('Data', file_name + '.mat'),
        os.path.join(root_dir, 'data', file_name + '.mat'),
        os.path.join(root_dir, 'Data', file_name + '.mat'),
    ]

    data_path = None
    for path in data_path_candidates:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            'Cannot find dataset. Tried:\n' + '\n'.join(data_path_candidates)
        )

    print(f'Loading data from: {data_path}')
    mat_data = scipy.io.loadmat(data_path)

    beta_all = mat_data['betas']
    gamma_all = mat_data['Gammas']
    phi_all = mat_data['Phii_cf'].transpose(0, 2, 1)
    rates_equal_solutions = mat_data['R_equal'][0]
    rates_frac_solutions = mat_data['R_frac'][0]
    rates_log_solutions = mat_data['R_log'][0]

    perm = np.random.RandomState(args.seed).permutation(beta_all.shape[0])
    train_idx = perm[:args.num_train]
    test_idx = perm[args.num_train: args.num_train + args.num_test]
    eval_idx = perm[-args.num_eval:]

    train_data, train_loader = build_homo_loader_qattention(
        beta_all[train_idx],
        gamma_all[train_idx],
        phi_all[train_idx],
        args.batch_size,
        isShuffle=True,
        device=device,
    )
    test_data, test_loader = build_homo_loader_qattention(
        beta_all[test_idx],
        gamma_all[test_idx],
        phi_all[test_idx],
        args.batch_size,
        isShuffle=False,
        device=device,
    )
    eval_data, eval_loader = build_homo_loader_qattention(
        beta_all[eval_idx],
        gamma_all[eval_idx],
        phi_all[eval_idx],
        args.num_eval,
        isShuffle=False,
        device=device,
    )

    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1]
    print(f'Node dim: {node_dim} | Edge dim: {edge_dim}')

    # ---------------------- QAttention-QGNN ----------------------
    aux_qubit = 1
    args.node_qubit = args.graphlet_size
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    q_dev = qml.device(args.q_dev, wires=n_qubits + aux_qubit)
    print(f'Quantum device: {n_qubits} qubit - {q_dev}')

    w_shapes_dict = {
        'inits': (args.num_ent_layers, 4),
        'strong': (args.num_ent_layers, 4),
        'update': (edge_qubit, args.num_ent_layers, 2 + aux_qubit, 3),
    }

    # qgnn_model = QGNN_DL_QAttention(
    #     q_dev=q_dev,
    #     w_shapes=w_shapes_dict,
    #     hidden_dim=args.hidden_channels,
    #     node_input_dim=node_dim,
    #     edge_input_dim=edge_dim,
    #     graphlet_size=args.node_qubit,
    #     hop_neighbor=args.num_gnn_layers,
    #     qattn_layers=args.qattn_layers,
    # ).to(device)

    qgnn_model = QGNN_DL_QAttention_Context(
        q_dev=q_dev,
        w_shapes=w_shapes_dict,
        hidden_dim=args.hidden_channels,
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        graphlet_size=args.node_qubit,
        hop_neighbor=args.num_gnn_layers,
        relation_filter=True,
        qattn_layers=args.qattn_layers,
        context_aggr=args.context_aggr,
    ).to(device)

    quantum_params, classical_params = [], []
    for name, param in qgnn_model.named_parameters():
        # qconvs = main graphlet PQC, attn.q_layer = QAttention PQC
        if 'qconvs' in name or 'attn.q_layer' in name:
            quantum_params.append(param)
        else:
            classical_params.append(param)

    qgnn_optimizer = torch.optim.Adam([
        {'params': classical_params, 'lr': args.lr_c},
        {'params': quantum_params, 'lr': args.lr_q},
    ])

    qgnn_scheduler = torch.optim.lr_scheduler.StepLR(
        qgnn_optimizer,
        step_size=max(1, args.step_size),
        gamma=args.gamma,
    )

    qgnn_all_rate = []
    qgnn_all_rate_test = []

    if qgnn_pretrain is not None:
        qgnn_model_filename = os.path.join(MODEL_DIR, qgnn_pretrain + '.pth')
        qgnn_model.load_state_dict(torch.load(qgnn_model_filename, map_location=device))
        print(f'Loaded pre-trained QAttention-QGNN from {qgnn_model_filename}.')
    else:
        start_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        print(f'\n ===={start_stamp}==== Training QAttention-QGNN on {device} ... ')
        start_time = time.time()
        eval_epochs = args.num_epochs // 10 if args.num_epochs // 10 else 1

        print(
            f'Training QAttention-QGNN '
            f'({args.graphlet_size}-node graphlet, '
            f'{args.num_gnn_layers} hops, '
            f'qattn_layers={args.qattn_layers}, '
            f'epochs={args.num_epochs})'
        )
        print(
            f'Equal Power rate: train {np.mean(rates_equal_solutions[train_idx]):.4f}, '
            f'test {np.mean(rates_equal_solutions[test_idx]):.4f}'
        )
        print(
            f'Log Approximation rate: train {np.mean(rates_log_solutions[train_idx]):.4f}, '
            f'test {np.mean(rates_log_solutions[test_idx]):.4f}'
        )

        for epoch in range(args.num_epochs):
            qgnn_model.train()
            train_loss = train_sumrate_homo(
                epoch / max(1, (2 * args.num_epochs // 3)),
                train_loader,
                qgnn_model,
                qgnn_optimizer,
                tau=tau,
                rho_p=rho_p,
                rho_d=rho_d,
                num_antenna=num_antenna,
                device=device,
            )

            qgnn_model.eval()
            with torch.no_grad():
                train_eval = eval_sumrate_homo(
                    train_loader,
                    qgnn_model,
                    tau=tau,
                    rho_p=rho_p,
                    rho_d=rho_d,
                    num_antenna=num_antenna,
                    device=device,
                )
                test_eval = eval_sumrate_homo(
                    test_loader,
                    qgnn_model,
                    tau=tau,
                    rho_p=rho_p,
                    rho_d=rho_d,
                    num_antenna=num_antenna,
                    device=device,
                )

            qgnn_all_rate.append(train_eval)
            qgnn_all_rate_test.append(test_eval)
            qgnn_scheduler.step()

            if epoch % eval_epochs == 0:
                print(
                    f'Epoch {epoch+1:03d}/{args.num_epochs} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Train Rate: {train_eval:.4f} | '
                    f'Test Rate: {test_eval:.4f}'
                )

        execution_time = time.time() - start_time
        print(f'Execution Time: {timedelta(seconds=execution_time)}')

        plt.figure(figsize=(6, 4), dpi=180)
        plt.plot(qgnn_all_rate, label='Training Rate', linewidth=2)
        plt.plot(qgnn_all_rate_test, label='Testing Rate', linewidth=2)
        plt.axhline(
            y=np.mean(rates_log_solutions[train_idx]),
            linewidth=2,
            color='r',
            linestyle='--',
            label='Training Log Approx.',
        )
        plt.axhline(
            y=np.mean(rates_log_solutions[test_idx]),
            linewidth=2,
            color='b',
            linestyle='--',
            label='Testing Log Approx.',
        )
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title('QAttention-QGNN Training Rate Curve', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(TRAIN_DIR, f'{timestamp}_qattn_qgnn.png')
        plt.savefig(save_path, dpi=300)
        print(f'Save training curve to {save_path}.')

        qgnn_model_filename = os.path.join(MODEL_DIR, f'{timestamp}_qattn_qgnn.pth')
        torch.save(qgnn_model.state_dict(), qgnn_model_filename)
        print(f'Save QAttention-QGNN to {qgnn_model_filename}.')

    # ---------------------- Centralized GNN benchmark ----------------------
    c_model = None
    if not args.skip_cen:
        c_model = HomoCfmMimoNet(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_channels=args.cen_hidden_channels,
            num_layers=args.cen_num_gnn_layers // 2,
        ).to(device)

        c_optimizer = torch.optim.AdamW(
            c_model.parameters(),
            lr=args.cen_lr,
            weight_decay=1e-4,
        )
        c_scheduler = torch.optim.lr_scheduler.StepLR(
            c_optimizer,
            step_size=max(1, args.num_epochs_cen // 10),
            gamma=0.8,
        )

        if c_pretrain is not None:
            model_filename = os.path.join(MODEL_DIR, c_pretrain + '.pth')
            c_model.load_state_dict(torch.load(model_filename, map_location=device))
            print(f'Loaded centralized GNN from {model_filename}.')
        else:
            start_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            print(f'\n ===={start_stamp}==== Training Centralized GNN benchmark ... ')
            start_time = time.time()
            all_rate = []
            all_rate_test = []
            eval_epochs_cen = args.num_epochs_cen // 10 if args.num_epochs_cen // 10 else 1

            print(
                f'Log Approximation rate: train {np.mean(rates_log_solutions[train_idx]):.4f}, '
                f'test {np.mean(rates_log_solutions[test_idx]):.4f}'
            )

            for epoch in range(args.num_epochs_cen):
                c_model.train()
                train_loss = train_sumrate_homo(
                    epoch / max(1, (2 * args.num_epochs_cen // 3)),
                    train_loader,
                    c_model,
                    c_optimizer,
                    tau=tau,
                    rho_p=rho_p,
                    rho_d=rho_d,
                    num_antenna=num_antenna,
                    device=device,
                )

                c_model.eval()
                with torch.no_grad():
                    train_eval = eval_sumrate_homo(
                        train_loader,
                        c_model,
                        tau=tau,
                        rho_p=rho_p,
                        rho_d=rho_d,
                        num_antenna=num_antenna,
                        device=device,
                    )
                    test_eval = eval_sumrate_homo(
                        test_loader,
                        c_model,
                        tau=tau,
                        rho_p=rho_p,
                        rho_d=rho_d,
                        num_antenna=num_antenna,
                        device=device,
                    )

                all_rate.append(train_eval)
                all_rate_test.append(test_eval)
                c_scheduler.step()

                if epoch % eval_epochs_cen == 0:
                    print(
                        f'Epoch {epoch+1:03d}/{args.num_epochs_cen} | '
                        f'Train Loss: {train_loss:.4f} | '
                        f'Train Rate: {train_eval:.4f} | '
                        f'Test Rate: {test_eval:.4f}'
                    )

            execution_time = time.time() - start_time
            print(f'Execution Time: {timedelta(seconds=execution_time)}')

            plt.figure(figsize=(6, 4), dpi=180)
            plt.plot(all_rate, label='Training Rate', linewidth=2)
            plt.plot(all_rate_test, label='Testing Rate', linewidth=2)
            plt.axhline(
                y=np.mean(rates_log_solutions[train_idx]),
                linewidth=2,
                color='r',
                linestyle='--',
                label='Training Log Approx.',
            )
            plt.axhline(
                y=np.mean(rates_log_solutions[test_idx]),
                linewidth=2,
                color='b',
                linestyle='--',
                label='Testing Log Approx.',
            )
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Rate', fontsize=12)
            plt.title('Centralized GNN Training Rate Curve', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(TRAIN_DIR, f'{timestamp}_cen.png')
            plt.savefig(save_path, dpi=300)
            print(f'Save centralized training curve to {save_path}.')

            model_filename = os.path.join(MODEL_DIR, f'{timestamp}_cen.pth')
            torch.save(c_model.state_dict(), model_filename)
            print(f'Save centralized GNN to {model_filename}.')

    # ---------------------- Evaluation - CDF ----------------------
    if args.eval_plot:
        print('Evaluation' + '=' * 20)

        qgnn_model.eval()
        if c_model is not None:
            c_model.eval()

        qgnn_rates_list = []
        gnn_rates_list = []
        all_one_rates_list = []

        for batch in eval_loader:
            batch = batch.to(device)

            with torch.no_grad():
                q_x = qgnn_model(batch)
                q_rates, one_rates = loss_function_sumrate_homo(
                    batch,
                    q_x,
                    tau=tau,
                    rho_p=rho_p,
                    rho_d=rho_d,
                    num_antenna=num_antenna,
                    eval_mode=True,
                )

                qgnn_rates_list.append(q_rates.detach().cpu().numpy().reshape(-1))
                all_one_rates_list.append(one_rates.detach().cpu().numpy().reshape(-1))

                if c_model is not None:
                    c_x = c_model(batch)
                    c_rates, _ = loss_function_sumrate_homo(
                        batch,
                        c_x,
                        tau=tau,
                        rho_p=rho_p,
                        rho_d=rho_d,
                        num_antenna=num_antenna,
                        eval_mode=True,
                    )
                    gnn_rates_list.append(c_rates.detach().cpu().numpy().reshape(-1))

        qgnn_rates = np.concatenate(qgnn_rates_list)
        all_one_rates = np.concatenate(all_one_rates_list)

        if len(gnn_rates_list) > 0:
            gnn_rates = np.concatenate(gnn_rates_list)
        else:
            gnn_rates = None
        rates_equal = rates_equal_solutions[eval_idx]
        rates_frac = rates_frac_solutions[eval_idx]
        rates_log = rates_log_solutions[eval_idx]

        max_candidates = [
            np.max(all_one_rates),
            np.max(qgnn_rates),
            np.max(rates_equal),
            np.max(rates_frac),
            np.max(rates_log),
        ]
        if gnn_rates is not None:
            max_candidates.append(np.max(gnn_rates))

        max_value = np.ceil(max(max_candidates) * 100) / 100

        if gnn_rates is not None:
            print(
                f'Sum rate avg: Centralized GNN {gnn_rates.mean():.2f} - '
                f'QAttention-QGNN {qgnn_rates.mean():.2f} - '
                f'{qgnn_rates.mean() * 100 / max(gnn_rates.mean(), 1e-12):.2f}%'
            )
        else:
            print(f'Sum rate avg: QAttention-QGNN {qgnn_rates.mean():.2f}')

        min_rate, max_rate = 0, max_value
        num_eval_actual = len(qgnn_rates)
        y_axis = np.linspace(0, 1, num_eval_actual + 2)
        def prep_cdf(x):
            x = np.asarray(x).reshape(-1)
            x = np.sort(x.copy())
            x = np.insert(x, 0, min_rate)
            x = np.insert(x, len(x), max_rate)
            return x
        qgnn_rates = prep_cdf(qgnn_rates)
        all_one_rates = prep_cdf(all_one_rates)
        rates_equal = prep_cdf(rates_equal)
        rates_frac = prep_cdf(rates_frac)
        rates_log = prep_cdf(rates_log)
        if gnn_rates is not None:
            gnn_rates = prep_cdf(gnn_rates)

        qgnn_rates = np.insert(qgnn_rates, 0, min_rate)
        qgnn_rates = np.insert(qgnn_rates, args.num_eval + 1, max_rate)
        all_one_rates = np.insert(all_one_rates, 0, min_rate)
        all_one_rates = np.insert(all_one_rates, args.num_eval + 1, max_rate)
        rates_equal = np.insert(rates_equal, 0, min_rate)
        rates_equal = np.insert(rates_equal, args.num_eval + 1, max_rate)
        rates_frac = np.insert(rates_frac, 0, min_rate)
        rates_frac = np.insert(rates_frac, args.num_eval + 1, max_rate)
        rates_log = np.insert(rates_log, 0, min_rate)
        rates_log = np.insert(rates_log, args.num_eval + 1, max_rate)
        if gnn_rates is not None:
            gnn_rates = np.insert(gnn_rates, 0, min_rate)
            gnn_rates = np.insert(gnn_rates, args.num_eval + 1, max_rate)

        plt.figure(figsize=(6, 4), dpi=180)
        if gnn_rates is not None:
            plt.plot(gnn_rates, y_axis, label='Centralized GNN', linewidth=2)
        plt.plot(qgnn_rates, y_axis, label='QAttention-QGNN', linewidth=2)
        plt.plot(rates_equal, y_axis, label='Equal Power', linewidth=2)
        plt.plot(rates_log, y_axis, label='Log Approx.', linewidth=2)
        plt.xlabel('Sum rate [bps/Hz]', {'fontsize': 16})
        plt.ylabel('Empirical CDF', {'fontsize': 16})
        plt.legend(fontsize=12)
        plt.grid()

        eval_path = os.path.join(EVAL_DIR, f'{timestamp}_qattn_eval.png')
        plt.savefig(eval_path, dpi=300, bbox_inches='tight')
        print(f'Save evaluation figure to {eval_path}.')
