import torch
import os
import numpy as np
import scipy.io
from itertools import combinations

from torch_geometric.data import Dataset, Data, HeteroData
from comm_utils import normalize_data


class cfHetGraphDataset(Dataset):
    def __init__(self, norm_losses, direct, cross, KM):
        self.norm_losses = norm_losses
        self.direct = torch.tensor(direct, dtype=torch.float32)
        self.cross = torch.tensor(cross, dtype=torch.float32)
        self.KM = KM
        self.num_samples, self.num_AP, self.num_UE = self.norm_losses.shape
        self.num_nodes = self.num_AP * self.num_UE
        self.adj, self.adj_t = self.build_edge_index()
        # Generate the dataset (graphs and labels)
        self.graphs = [self.create_graph(idx) for idx in range(self.num_samples)]
        
    def build_edge_index(self):
        adj, adj_t = [], []
        for k in range(self.KM[0]):  # UE
            for m in range(self.KM[1]):  # AP
                adj.append([k, m])
                adj_t.append([m, k])
        return adj, adj_t
    
    def create_graph(self, idx):
        H = self.norm_losses[idx]
        edge_feat = H.reshape((self.KM[0] * self.KM[1], 1), order='F')
        edge_feat = np.concatenate([edge_feat, np.ones_like(edge_feat)], axis=-1)
        edge_feat = torch.tensor(edge_feat, dtype=torch.float32)

        edge_index = torch.tensor(self.adj, dtype=torch.long).T.contiguous()
        edge_index_t = torch.tensor(self.adj_t, dtype=torch.long).T.contiguous()

        ue_feat = torch.ones((self.KM[0], 1), dtype=torch.float32)
        ap_feat = torch.ones((self.KM[1], 1), dtype=torch.float32)

        edge_sum_ue = edge_feat.view(self.KM[0], self.KM[1], 2).sum(dim=1)
        edge_sum_ap = edge_feat.view(self.KM[0], self.KM[1], 2).sum(dim=0).reshape(self.KM[1], 2)

        data = HeteroData()
        data['UE'].x = ue_feat
        data['AP'].x = ap_feat
        data['UE', 'com-by', 'AP'].edge_index = edge_index
        data['UE', 'com-by', 'AP'].edge_attr = edge_feat
        data['AP', 'com', 'UE'].edge_index = edge_index_t
        data['AP', 'com', 'UE'].edge_attr = edge_feat
        data['UE'].edge_sum = edge_sum_ue
        data['AP'].edge_sum = edge_sum_ap
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.graphs[idx], self.direct[idx], self.cross[idx]
        

class cfGraphDataset(Dataset):
    def __init__(self, norm_losses, direct, cross, KM):
        self.norm_losses = norm_losses
        self.direct = torch.tensor(direct, dtype=torch.float32)
        self.cross = torch.tensor(cross, dtype=torch.float32)
        self.KM = KM

        self.num_samples, self.num_AP, self.num_UE = self.norm_losses.shape

        self.num_nodes = self.num_AP * self.num_UE

        # Generate the dataset (graphs and labels)
        self.graphs = [self.create_graph(idx) for idx in range(self.num_samples)]

    # def create_graph(self, idx):
    #     H = self.norm_losses[idx]
    #     x = torch.tensor(H.flatten()[:,None], dtype=torch.float32)

    #     # Fully connected graph
    #     edge_index = torch.tensor([
    #         [i, j] for i in range(self.num_nodes) for j in range(i+1, self.num_nodes) #if i != j
    #         # [i, j] for i in range(self.num_nodes) for j in range(self.num_nodes) if i != j
    #     ], dtype=torch.long).T  # shape: [2, num_edges]

    #     node_indices = np.arange(self.num_nodes)
    #     row_indices, col_indices = np.unravel_index(node_indices, (self.num_AP, self.num_UE))
    #     node_feat = H[row_indices, col_indices]
    #     edge_attr_mtx = H[row_indices[:, None], col_indices[None, :]]
    #     edge_attr_mtx = torch.tensor(edge_attr_mtx, dtype=torch.float32)
    #     # edge_attr = edge_attr_mtx[edge_index[0], edge_index[1]][:,None]
    #     edge_attr = torch.cat([edge_attr_mtx[edge_index[0], edge_index[1]][:,None], edge_attr_mtx[edge_index[1], edge_index[0]][:,None]], dim=-1)

    #     # y = torch.tensor(self.labels[idx], dtype=torch.float32)

    #     data = Data(
    #         x=x,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr,
    #         direct = self.direct[idx],
    #         cross = self.cross[idx],
    #         # y=y
    #     )
    #     return data
    
    def create_graph(self, idx):
        H = self.norm_losses[idx]
        edge_index = []
        edge_attr = []
        
        node_ft = []

        # node_ft = H.flatten() 
        ## TODO: Limited edge
        # for ap in range(self.num_AP):
        #     node_ids = [k + ap * self.num_UE for k in range(self.num_UE)]
        #     for i, j in combinations(node_ids, 2):
        #         edge_index.append([i, j])
        #         # edge_attr.append([node_ft[i], node_ft[j]])
        #         edge_index.append([j, i])
        #         # edge_attr.append([node_ft[j], node_ft[i]])
        # # for ue in range(self.num_UE):
        # #     node_ids = [ue + m * self.num_UE for m in range(self.num_AP)]
        # #     for i, j in combinations(node_ids, 2):
        # #         edge_index.append([i, j])
        # #         edge_attr.append([node_ft[i], node_ft[j]])
        #         # edge_index.append([j, i])
        #         # edge_attr.append([node_ft[j], node_ft[i]])
        # edge_index = torch.tensor(edge_index, dtype=torch.long).T
        # # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        # edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)

        ## TODO: Using flag for same UE or AP
        for m in range(self.num_AP):
            for k in range(self.num_UE):
                node_ft.append([
                    H[m, k],
                    np.sum(H[m, :]),
                    np.sum(H[:, k]),
                    # H[m, k] / (np.sum(H[m, :]) + 1e-6),
                    # H[m, k] / (np.sum(H[:, k]) + 1e-6),
                ])
                node_id = m * self.num_UE + k
                for m2 in range(self.num_AP):
                    for k2 in range(self.num_UE):
                        if m == m2 or k == k2:
                            node_id2 = m2 * self.num_UE + k2
                            edge_index.append([node_id, node_id2])
                            edge_attr.append([
                                H[m, k2], H[m2, k], 
                                int(m == m2) - int(k == k2)
                            ])
        x = torch.tensor(node_ft, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        

        
        ## TODO: Fully connected
        ###
        # rows, cols = np.triu_indices(self.num_nodes, k=1)  
        # m1, k1 = divmod(rows, self.num_UE)
        # m2, k2 = divmod(cols, self.num_UE)
        # edge_attr = np.stack([
        #     # H[m1, k1],    
        #     # H[m2, k2],    
        #     H[m2, k1],    
        #     H[m1, k2],    
        # ], axis=1)  
        # edge_index = np.stack([rows, cols], axis=0) 
        # edge_index = torch.tensor(edge_index, dtype=torch.long)
        # edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        ####
        
        # x = torch.tensor(node_ft[:, None], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            direct = self.direct[idx],
            cross = self.cross[idx],
            # y=y
        )
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.graphs[idx], self.direct[idx], self.cross[idx]



def load_cf_dataset(train_path, test_path, training_sam=100, testing_sam=50):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset not found at {train_path}. Please check the path.")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset not found at {test_path}. Please check the path.")
    data = scipy.io.loadmat(train_path)
    beta_train = data['betas'][:training_sam]
    direct_train = data['directs'][:training_sam]
    cross_train = data['corsses'][:training_sam].transpose(0,2,1)
    
    test_data = scipy.io.loadmat(test_path)
    beta_test = test_data['betas'][:testing_sam]
    direct_test = test_data['directs'][:testing_sam]
    cross_test = test_data['corsses'][:testing_sam].transpose(0,2,1)
    opt_rate = test_data['R_cf_opt_min'][:,:testing_sam]
    
    norm_train_losses, norm_test_losses = normalize_data(beta_train**(1/2), beta_test**(1/2) )
    
    train_K, train_M = beta_train.shape[2], beta_train.shape[1]
    test_K, test_M = beta_test.shape[2], beta_test.shape[1]


    return norm_train_losses, direct_train, cross_train, (train_K, train_M),\
            norm_test_losses, direct_test, cross_test, (test_K, test_M), opt_rate
            
            