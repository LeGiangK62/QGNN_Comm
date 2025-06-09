import torch
import os
import numpy

from torch_geometric.data import Dataset, Data
from comm_utils import generate_wGaussian, np_sum_rate_all


class d2dGraphDataset(Dataset):
    def __init__(self, num_samples, num_D2D, p_max, n0, seed=1309):
        self.num_samples = num_samples
        self.num_D2D = num_D2D
        self.p_max = p_max  # Transmit power
        self.n0 = n0  # Noise power
        self.seed = seed

        # Generate the dataset (graphs and labels)
        self.channels, self.w_sumrate, self.objs, self.labels = self.generate_data()
        self.graphs = [self.create_graph(idx) for idx in range(self.num_samples)]


    def generate_data(self):
        channel_matrices, power_matrices, weight_matrices, _ = generate_wGaussian(
            self.num_D2D, self.num_samples, seed=self.seed, var_noise = self.n0
        )

        wmmse_sumrate = np_sum_rate_all(channel_matrices,power_matrices,weight_matrices,self.n0)[:,None]

        print(f'Benchmark WMMSE: {numpy.mean(wmmse_sumrate)}')
        return channel_matrices.transpose(0,2,1), weight_matrices, wmmse_sumrate, power_matrices

    def create_graph(self, idx):
        H = self.channels[idx] 
        W = self.w_sumrate[idx] #.repeat(self.num_D2D, 1)
        x1 = torch.tensor(H.diagonal()[:, None], dtype=torch.float32)
        x2 = torch.tensor(W[:, None], dtype=torch.float32)
        x = torch.cat([x1, x2], dim=1)  # shape: [num_D2D, 2]
        # Fully connected graph
        edge_index = torch.tensor([
            [i, j] for i in range(self.num_D2D) for j in range(self.num_D2D) if i != j
        ], dtype=torch.long).T  # shape: [2, num_edges]

        edge_attr = torch.tensor(H[edge_index[0], edge_index[1]][:, None], dtype=torch.float32)

        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.graphs[idx]

