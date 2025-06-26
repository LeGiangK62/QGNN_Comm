import torch
import random
from collections import defaultdict
import pickle

# from torchmetrics.classification import MulticlassF1Score

from comm_utils import rate_loss


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

def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    total_sample = 0
    for data, direct, cross in train_loader:
        bs = data.num_graphs
        M = direct.shape[1]
        K = data.x_dict['AP'].shape[0] // bs
        optimizer.zero_grad()

        output = model(data.x_dict, data.edge_attr_dict, data.edge_index_dict, data.batch_dict) # .reshape(bs, -1)
        power = output.reshape(bs, M)
        loss = rate_loss(power, direct, cross)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bs
        total_sample += bs

    return total_loss / total_sample

def test(model, test_loader):
    model.eval()
    total_loss = 0
    total_sample = 0
    with torch.no_grad():
        for data, direct, cross in test_loader:
            bs = data.num_graphs
            M = direct.shape[1]
            K = data.x_dict['AP'].shape[0] // bs
            # optimizer.zero_grad()

            output = model(data.x_dict, data.edge_attr_dict, data.edge_index_dict, data.batch_dict) # .reshape(bs, -1)
            # output = output.reshape(bs,-1)
            power = output.reshape(bs, M)
            loss = rate_loss(power, direct, cross)
            # loss.backward()

            total_loss += loss.item() * bs
            total_sample += bs

    return total_loss / total_sample


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

