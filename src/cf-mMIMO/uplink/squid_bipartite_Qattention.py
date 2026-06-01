import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# =========================================================
# Basic MLP
# =========================================================
class MLP(nn.Module):
    def __init__(self, dims, act="leaky_relu", norm=None, dropout=0.0):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                if norm == "batch_norm":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif norm == "layer_norm":
                    layers.append(nn.LayerNorm(dims[i + 1]))

                if act == "relu":
                    layers.append(nn.ReLU())
                elif act == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.2))
                elif act == "gelu":
                    layers.append(nn.GELU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# Quantum utilities
# =========================================================
def required_amplitude_qubits(flat_dim):
    return math.ceil(math.log2(flat_dim))


def make_amplitude_squid_qnode(
    n_qubits,
    flat_dim,
    n_layers=1,
    q_device="default.qubit",
):
    target_dim = 2 ** n_qubits

    if target_dim < flat_dim:
        raise ValueError(
            f"n_qubits={n_qubits} gives 2^n={target_dim}, "
            f"but flat_dim={flat_dim}. Need at least "
            f"{required_amplitude_qubits(flat_dim)} qubits."
        )

    dev = qml.device(q_device, wires=n_qubits)

    weight_shapes = {
        "weights": (n_layers, n_qubits, 3),
    }

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        x = inputs

        if flat_dim < target_dim:
            x = F.pad(x, (0, target_dim - flat_dim))
        elif flat_dim > target_dim:
            x = x[:target_dim]

        qml.AmplitudeEmbedding(
            features=x,
            wires=list(range(n_qubits)),
            normalize=True,
            pad_with=0.0,
        )

        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RX(weights[layer, q, 0], wires=q)
                qml.RY(weights[layer, q, 1], wires=q)
                qml.RZ(weights[layer, q, 2], wires=q)

            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

            if n_qubits > 1:
                qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit, weight_shapes


class AmplitudeSQUIDQuantumLayer(nn.Module):
    def __init__(
        self,
        flat_dim,
        n_qubits=None,
        n_layers=1,
        q_device="default.qubit",
    ):
        super().__init__()

        self.flat_dim = flat_dim

        if n_qubits is None:
            n_qubits = required_amplitude_qubits(flat_dim)

        self.n_qubits = n_qubits

        qnode, weight_shapes = make_amplitude_squid_qnode(
            n_qubits=n_qubits,
            flat_dim=flat_dim,
            n_layers=n_layers,
            q_device=q_device,
        )

        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, q_inputs_batch):
        outs = []
        for i in range(q_inputs_batch.shape[0]):
            outs.append(self.q_layer(q_inputs_batch[i]))
        return torch.stack(outs, dim=0)


class QNNAttentionScore(nn.Module):
    def __init__(self, in_dim, n_qubits=None, n_layers=1, q_device="default.qubit"):
        super().__init__()

        if n_qubits is None:
            n_qubits = math.ceil(math.log2(in_dim))

        self.in_dim = in_dim
        self.n_qubits = n_qubits
        target_dim = 2 ** n_qubits

        dev = qml.device(q_device, wires=n_qubits)

        weight_shapes = {
            "weights": (n_layers, n_qubits, 3)
        }

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            x = inputs

            if in_dim < target_dim:
                x = F.pad(x, (0, target_dim - in_dim))
            else:
                x = x[:target_dim]

            qml.AmplitudeEmbedding(
                features=x,
                wires=list(range(n_qubits)),
                normalize=True,
                pad_with=0.0,
            )

            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RX(weights[l, q, 0], wires=q)
                    qml.RY(weights[l, q, 1], wires=q)
                    qml.RZ(weights[l, q, 2], wires=q)

                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            return qml.expval(qml.PauliZ(0))

        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, edge_inputs):
        scores = []
        for i in range(edge_inputs.shape[0]):
            scores.append(self.q_layer(edge_inputs[i]))
        return torch.stack(scores, dim=0).view(-1)

class BipartiteSQUIDGNNPowerControl_QAttention(nn.Module):
    """
    Two-phase bipartite SQUID-GNN with learned hard top-k sampling.

    Changes:
    - Remove AP/UE positional embedding.
    - Use edge-conditioned node initialization by SUM aggregation.
    - Use learned hard top-k sampler for UE->AP and AP->UE.
    """

    def __init__(
        self,
        ap_in=1,
        ue_in=1,
        edge_dim=1,
        hidden_dim=32,
        pqc_dim=2,
        top_l_aps=3,
        top_l_ues=3,
        n_qubits_amp=None,
        q_layers=1,
        mp_layers=1,
        q_device="default.qubit",
        dropout=0.1,
        residual_scale=0.5,
        noise_std=0.01,
        output_temperature=2.0,
        output_bias_init=-1.0,
        use_softmax_power=False,
        edge_scale=5.0,
        epsilon_random=0.0,
    ):
        super().__init__()

        self.ap_in = ap_in
        self.ue_in = ue_in
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.pqc_dim = pqc_dim

        self.top_l_aps = top_l_aps
        self.top_l_ues = top_l_ues
        self.mp_layers = mp_layers

        self.residual_scale = residual_scale
        self.noise_std = noise_std
        self.output_temperature = output_temperature
        self.use_softmax_power = use_softmax_power
        self.edge_scale = edge_scale
        self.epsilon_random = epsilon_random

        self.flat_dim = (2 * top_l_aps + 1) * pqc_dim

        if n_qubits_amp is None:
            n_qubits_amp = required_amplitude_qubits(self.flat_dim)

        self.n_qubits_amp = n_qubits_amp
        self.q_out_dim = n_qubits_amp

        self.ap_encoder = MLP([ap_in, hidden_dim, pqc_dim], dropout=0.0)
        self.ue_encoder = MLP([ue_in, hidden_dim, pqc_dim], dropout=0.0)
        self.edge_encoder = MLP([edge_dim, hidden_dim, pqc_dim], dropout=0.0)

        score_in_dim = 3 * pqc_dim

        self.ue2ap_score = QNNAttentionScore(
            in_dim=score_in_dim,
            n_layers=1,
            q_device=q_device,
        )

        self.ap2ue_score = QNNAttentionScore(
            in_dim=score_in_dim,
            n_layers=1,
            q_device=q_device,
        )

        self.ue_to_ap_msg = nn.ModuleList([
            MLP([3 * pqc_dim, hidden_dim, pqc_dim], dropout=dropout)
            for _ in range(mp_layers)
        ])

        self.ap_update = nn.ModuleList([
            MLP([2 * pqc_dim, hidden_dim, pqc_dim], dropout=dropout)
            for _ in range(mp_layers)
        ])

        self.q_layers = nn.ModuleList([
            AmplitudeSQUIDQuantumLayer(
                flat_dim=self.flat_dim,
                n_qubits=n_qubits_amp,
                n_layers=q_layers,
                q_device=q_device,
            )
            for _ in range(mp_layers)
        ])

        self.ue_update = nn.ModuleList([
            MLP(
                [pqc_dim + self.q_out_dim, hidden_dim, pqc_dim],
                dropout=dropout,
            )
            for _ in range(mp_layers)
        ])

        self.power_head = MLP(
            [pqc_dim, hidden_dim, hidden_dim, 1],
            dropout=dropout,
        )

        self._init_power_head(output_bias_init)

    def _init_power_head(self, bias):
        last = self.power_head.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight, gain=0.1)
            nn.init.constant_(last.bias, bias)

    def _get_edge_key(self, edge_index_dict):
        preferred = ("ap", "serves", "ue")
        if preferred in edge_index_dict:
            return preferred

        for key in edge_index_dict.keys():
            if key[0] == "ap" and key[-1] == "ue":
                return key

        raise KeyError("Cannot find AP->UE edge type")

    def _maybe_random_or_topk(self, score, max_neighbors):
        n = score.numel()
        device = score.device

        if n <= max_neighbors:
            return torch.arange(n, device=device)

        if self.training and self.epsilon_random > 0:
            if torch.rand((), device=device) < self.epsilon_random:
                return torch.randperm(n, device=device)[:max_neighbors]

        return torch.topk(score, k=max_neighbors, largest=True).indices

    def _edge_conditioned_init(self, ap_h, ue_h, edge_h, edge_index):
        src_ap = edge_index[0].long()
        dst_ue = edge_index[1].long()

        # SUM aggregation, no degree normalization
        ue_edge_aggr = torch.zeros_like(ue_h)
        ue_edge_aggr = ue_edge_aggr.index_add(0, dst_ue, edge_h)

        ap_edge_aggr = torch.zeros_like(ap_h)
        ap_edge_aggr = ap_edge_aggr.index_add(0, src_ap, edge_h)

        ue_h = ue_h + ue_edge_aggr
        ap_h = ap_h + ap_edge_aggr

        return ap_h, ue_h

    def _compute_edge_scores(self, ap_h, ue_h, edge_h, edge_index):
        src_ap = edge_index[0].long()
        dst_ue = edge_index[1].long()

        score_input = torch.cat(
            [ap_h[src_ap], ue_h[dst_ue], edge_h],
            dim=-1,
        )

        ue2ap_score = self.ue2ap_score(score_input)
        ap2ue_score = self.ap2ue_score(score_input)

        return ue2ap_score, ap2ue_score

    def _update_ap_from_ues(
        self,
        ap_h,
        ue_h,
        edge_h,
        edge_index,
        ue2ap_score,
        layer_idx,
    ):
        src_ap = edge_index[0].long()
        dst_ue = edge_index[1].long()

        device = ap_h.device
        num_aps = ap_h.shape[0]

        sampled_ap_ids = []
        sampled_ue_ids = []
        sampled_edge_ids = []

        for ap_id in range(num_aps):
            mask = src_ap == ap_id
            edge_ids = torch.nonzero(mask, as_tuple=False).view(-1)

            if edge_ids.numel() == 0:
                continue

            local_score = ue2ap_score[edge_ids]
            local_idx = self._maybe_random_or_topk(
                local_score,
                self.top_l_ues,
            )

            edge_ids = edge_ids[local_idx]
            ue_ids = dst_ue[edge_ids]

            sampled_edge_ids.append(edge_ids)
            sampled_ue_ids.append(ue_ids)
            sampled_ap_ids.append(
                torch.full(
                    (edge_ids.numel(),),
                    ap_id,
                    dtype=torch.long,
                    device=device,
                )
            )

        if len(sampled_edge_ids) == 0:
            return ap_h

        sampled_edge_ids = torch.cat(sampled_edge_ids, dim=0)
        sampled_ue_ids = torch.cat(sampled_ue_ids, dim=0)
        sampled_ap_ids = torch.cat(sampled_ap_ids, dim=0)

        msg_in = torch.cat(
            [
                ap_h[sampled_ap_ids],
                ue_h[sampled_ue_ids],
                edge_h[sampled_edge_ids],
            ],
            dim=-1,
        )

        msg = self.ue_to_ap_msg[layer_idx](msg_in)

        ap_aggr = torch.zeros_like(ap_h)
        ap_aggr = ap_aggr.index_add(0, sampled_ap_ids, msg)

        upd = self.ap_update[layer_idx](
            torch.cat([ap_h, ap_aggr], dim=-1)
        )

        ap_h = (1.0 - self.residual_scale) * ap_h + self.residual_scale * upd

        return ap_h

    def _build_ue_centered_q_inputs_batch(
        self,
        ap_h,
        ue_h,
        edge_h,
        edge_index,
        ap2ue_score,
    ):
        device = ue_h.device
        dtype = ue_h.dtype

        num_ues = ue_h.shape[0]

        src_ap = edge_index[0].long()
        dst_ue = edge_index[1].long()

        batched_inputs = []
        centers = []

        for ue_id in range(num_ues):
            mask = dst_ue == ue_id
            edge_ids = torch.nonzero(mask, as_tuple=False).view(-1)

            if edge_ids.numel() == 0:
                continue

            local_score = ap2ue_score[edge_ids]
            local_idx = self._maybe_random_or_topk(
                local_score,
                self.top_l_aps,
            )

            edge_ids = edge_ids[local_idx]
            ap_ids = src_ap[edge_ids]

            n_selected = edge_ids.numel()

            e_feat = torch.zeros(
                (self.top_l_aps, self.pqc_dim),
                device=device,
                dtype=dtype,
            )

            a_feat = torch.zeros(
                (self.top_l_aps, self.pqc_dim),
                device=device,
                dtype=dtype,
            )

            e_feat[:n_selected] = edge_h[edge_ids]
            a_feat[:n_selected] = ap_h[ap_ids]

            center_feat = ue_h[ue_id].view(1, self.pqc_dim)

            sample_inputs = torch.cat(
                [e_feat, center_feat, a_feat],
                dim=0,
            )

            batched_inputs.append(sample_inputs.flatten())
            centers.append(torch.tensor(ue_id, device=device, dtype=torch.long))

        if len(batched_inputs) == 0:
            return None, None

        q_inputs_batch = torch.stack(batched_inputs, dim=0)
        center_ids = torch.stack(centers, dim=0)

        return q_inputs_batch, center_ids

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        edge_key = self._get_edge_key(edge_index_dict)

        ap_x = x_dict["ap"].float()
        ue_x = x_dict["ue"].float()

        edge_index = edge_index_dict[edge_key]
        edge_attr = edge_attr_dict[edge_key].float()

        ap_h = self.ap_encoder(ap_x)
        ue_h = self.ue_encoder(ue_x)

        edge_h = self.edge_encoder(edge_attr)
        edge_h = torch.tanh(edge_h / self.edge_scale)

        # Practical initialization:
        # AP/UE node features remain constant,
        # diversity comes only from observed AP-UE edge information.
        ap_h, ue_h = self._edge_conditioned_init(
            ap_h=ap_h,
            ue_h=ue_h,
            edge_h=edge_h,
            edge_index=edge_index,
        )

        if self.training and self.noise_std > 0:
            ap_h = ap_h + self.noise_std * torch.randn_like(ap_h)
            ue_h = ue_h + self.noise_std * torch.randn_like(ue_h)

        for layer_idx in range(self.mp_layers):
            ue2ap_score, ap2ue_score = self._compute_edge_scores(
                ap_h=ap_h,
                ue_h=ue_h,
                edge_h=edge_h,
                edge_index=edge_index,
            )

            ap_h = self._update_ap_from_ues(
                ap_h=ap_h,
                ue_h=ue_h,
                edge_h=edge_h,
                edge_index=edge_index,
                ue2ap_score=ue2ap_score,
                layer_idx=layer_idx,
            )

            # Recompute AP->UE score after AP update
            _, ap2ue_score = self._compute_edge_scores(
                ap_h=ap_h,
                ue_h=ue_h,
                edge_h=edge_h,
                edge_index=edge_index,
            )

            q_inputs_batch, center_ids = self._build_ue_centered_q_inputs_batch(
                ap_h=ap_h,
                ue_h=ue_h,
                edge_h=edge_h,
                edge_index=edge_index,
                ap2ue_score=ap2ue_score,
            )

            if q_inputs_batch is None:
                continue

            q_out = self.q_layers[layer_idx](q_inputs_batch)

            upd_in = torch.cat([ue_h[center_ids], q_out], dim=-1)
            upd = self.ue_update[layer_idx](upd_in)

            updates_node = torch.zeros_like(ue_h)
            updates_node = updates_node.index_add(0, center_ids, upd)

            ue_h = (1.0 - self.residual_scale) * ue_h + self.residual_scale * updates_node

        logits = self.power_head(ue_h).squeeze(-1)

        if self.use_softmax_power:
            eta = torch.softmax(logits, dim=0) * ue_h.shape[0]
            eta = torch.clamp(eta, 0.0, 1.0)
        else:
            eta = torch.sigmoid(logits / self.output_temperature)

        return eta