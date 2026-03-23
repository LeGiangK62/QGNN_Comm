import torch
import random
from collections import defaultdict
import pickle

# from torchmetrics.classification import MulticlassF1Score


def star_subgraph(adjacency_matrix, subgraph_size=4):
    num_nodes = adjacency_matrix.shape[0]
    subgraph_indices = []
    uncovered_neighbors = set(range(num_nodes))  # All nodes should be covered as neighbors at least once

    leaf_counts = defaultdict(int)

    seed_nodes = list(range(num_nodes))
    random.shuffle(seed_nodes)

    for center_node in seed_nodes:
        neighbors = [i for i in range(num_nodes) if adjacency_matrix[center_node, i] != 0 and i != center_node]
        k = subgraph_size - 1

        candidates = neighbors  # Already excludes center node

        # Case 1: Not enough neighbors → take all of them
        if len(candidates) <= k:
            sampled_neighbors = candidates

        else:
            available_new = list(set(candidates) & uncovered_neighbors)

            # Case 2a: enough new nodes → sample from them
            if len(available_new) >= k:
                sampled_neighbors = random.sample(available_new, k)

            # Case 2b: not enough new nodes → take all + fill from candidates
            else:
                sampled_neighbors = available_new
                remaining_k = k - len(sampled_neighbors)
                remaining_pool = list(set(candidates) - set(sampled_neighbors))
                remaining_pool.sort(key=lambda x: leaf_counts[x])

                sampled_neighbors += remaining_pool[:remaining_k]

        # Update uncovered neighbor set
        uncovered_neighbors -= set(sampled_neighbors)
        for node in sampled_neighbors:
            leaf_counts[node] += 1

        # Add center + its sampled neighbors
        subgraph = [center_node] + sampled_neighbors
        subgraph_indices.append(subgraph)

    return subgraph_indices

def train(model, train_loader, optimizer, var, pmax):
    model.train()
    total_loss = 0
    total_rate = 0
    total_graph = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data in train_loader:    
        optimizer.zero_grad()

        labels = data.y # torch.Size([320])
        bs = data.num_graphs
        channel_matrices = data.x[:,0].reshape(bs,-1)
        weight_sumrates = data.x[:,1].reshape(bs,-1)
        output = model(data.x, data.edge_attr, data.edge_index, data.batch).float() #torch.Size([320, 1])
        output = output.reshape(bs,-1,1) * pmax

        # Get raw data
        all_channel_matrices = get_all_channel(data.x, data.edge_attr, bs)

        loss = sum_weighted_rate(all_channel_matrices, output, weight_sumrates, var)

        # loss = sum_weighted_rate(channel_matrices, output, weight_sumrates, var)
        loss = torch.neg(loss)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            power_new = output #quantize_output(output)
            sum_rate = sum_weighted_rate(all_channel_matrices, power_new, weight_sumrates, var)
        total_rate += sum_rate.item()
        total_loss += loss.item()
        total_graph += data.num_graphs

    return total_loss / total_graph, total_rate / total_graph


def test(model, test_loader, var, pmax):
    model.eval()
    total_loss = 0
    total_rate = 0
    total_graph = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in test_loader: 
            labels = data.y # torch.Size([320])
            bs = data.num_graphs
            channel_matrices = data.x[:,0].reshape(bs,-1)
            weight_sumrates = data.x[:,1].reshape(bs,-1)
            output = model(data.x, data.edge_attr, data.edge_index, data.batch).float() #torch.Size([320, 1])
            output = output.reshape(bs,-1,1) * pmax
            all_channel_matrices = get_all_channel(data.x, data.edge_attr, bs)

            sum_rate = sum_weighted_rate(all_channel_matrices, output, weight_sumrates, var)
            total_rate += sum_rate.item()
            total_graph += data.num_graphs

    return total_rate / total_graph


# def quantize_output(output, num_levels=4):
#     levels = torch.linspace(0, 1, steps=num_levels, device=output.device)
#     # Find closest level for each output value
#     quantized_output = torch.zeros_like(output)
#     for i in range(output.shape[0]):
#         quantized_output[i] = levels[torch.argmin(torch.abs(levels - output[i]))]
#     return quantized_output


# def sum_weighted_rate_1D(h, p, w, n0):

#     all_signal = torch.square(h * p.view(-1, 1))
#     des_signal = torch.diag(all_signal)
#     rx_signal = torch.sum(all_signal, dim=0)
#     inteference = rx_signal - des_signal + n0

#     sinr = des_signal/inteference
#     w_sumrate = torch.log2(1 + sinr * w)
#     return torch.sum(w_sumrate)


def get_all_channel(x, edge_attr, num_graph):
    B = num_graph
    K = x.shape[0]//num_graph
    direct_channel_matrices = x[:,0].reshape(B, K, -1)
    edge_attr = edge_attr.reshape(B, K * (K - 1), 1)
    dev = direct_channel_matrices.device
    dtype = direct_channel_matrices.dtype
    all_channel_matrices = torch.empty((B, K, K), dtype=dtype, device=dev)
    diag_mask = torch.eye(K, dtype=torch.bool, device=dev)
    all_channel_matrices[:, diag_mask] = direct_channel_matrices.squeeze(-1)
    all_channel_matrices[:, ~diag_mask] = edge_attr.squeeze(-1)

    return all_channel_matrices

def sum_weighted_rate_incorrect(h, p, w, n0):
    # n0 = 1/10**(n0/10)
    all_signal = torch.square(h.unsqueeze(1) * p)  # shape: (B, N, N)
    des_signal = torch.diagonal(all_signal, dim1=1, dim2=2)
    rx_signal = torch.sum(all_signal, dim=1)
    interference = rx_signal - des_signal + n0
    sinr = des_signal / interference
    w_sumrate = torch.log2(1 + sinr) * w # Fix weighted sum rate?
    return torch.sum(w_sumrate)


def sum_weighted_rate(h, p, w, n0):
    # n0 = 1/10**(n0/10)
    all_signal = torch.square(h * p)  # shape: (B, N, N)
    des_signal = torch.diagonal(all_signal, dim1=1, dim2=2)
    rx_signal = torch.sum(all_signal, dim=1)
    interference = rx_signal - des_signal + n0
    sinr = des_signal / interference
    w_sumrate = torch.log2(1 + sinr) * w # Fix weighted sum rate?
    return torch.sum(w_sumrate)


def save_checkpoint(model, optimizer, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, save_path="best_model.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

def smooth(values, weight=0.9):  # weight closer to 1 = smoother
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed