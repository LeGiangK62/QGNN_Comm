import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.data import Data

from comm_utils import power_from_raw, variance_calculate, rate_calculation, component_calculate, rate_from_component
# =====================================================================
# Graph construction & data loaders (centralized loaders for sum-rate)
# =====================================================================

def full_het_graph(
        beta_single_sample, gamma_single_sample, 
        label_single_all, phi_single_sample, 
        ap_id=None, sample_id=None, 
        
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_AP, num_UE = beta_single_sample.shape

    # Creating node features (random values for AP and UE nodes)
    ap_features = np.ones((num_AP, 1), dtype=np.float32)   # np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    ue_features = phi_single_sample  # Random feature for UE nodes (dim 1)
    # ue_features_dummy = np.ones((num_UE, 3), dtype=np.float32)   # np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    # ue_features = np.concatenate([ue_features, ue_features_dummy], axis=1)
    
    # Concatenate features for both AP and UE nodes
    x_ap = torch.tensor(ap_features, dtype=torch.float32).to(device)
    x_ue = torch.tensor(ue_features, dtype=torch.float32).to(device)

    # Define edges (connect AP to all UEs in a bipartite manner)
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []

    for ap_idx in range(num_AP):
        for ue_idx in range(num_UE):
            edge_index_ap_down_ue.append([ap_idx, ue_idx])  # AP (0) to UE (ue_idx)
            # edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)
    
    for ue_idx in range(num_UE):
        for ap_idx in range(num_AP):
            edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap = torch.tensor(edge_index_ue_up_ap, dtype=torch.long).t().contiguous().to(device)

    # edge_attr_ap_to_ue = torch.tensor(beta_single_sample.reshape(-1, 1), dtype=torch.float32).to(device)
    # edge_attr_ue_up_ap = torch.tensor(beta_single_sample.T.reshape(-1, 1), dtype=torch.float32).to(device)
        
    beta_up = beta_single_sample.reshape(-1, 1)
    gamma_up = gamma_single_sample.reshape(-1, 1)
    edge_attr_ap_to_ue = np.concatenate((beta_up, gamma_up), axis=1)
    edge_attr_ap_to_ue = torch.tensor(edge_attr_ap_to_ue, dtype=torch.float32).to(device)
    
    
    beta_down = beta_single_sample.T.reshape(-1, 1)
    gamma_down = gamma_single_sample.T.reshape(-1, 1)
    edge_attr_ue_up_ap = np.concatenate((beta_down, gamma_down), axis=1)
    edge_attr_ue_up_ap = torch.tensor(edge_attr_ue_up_ap, dtype=torch.float32).to(device)   
    # Create the heterogeneous graph data
    data = HeteroData()
    data['AP'].x = x_ap
    data['UE'].x = x_ue
    data['AP', 'down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'down', 'UE'].edge_attr = edge_attr_ap_to_ue
    data['UE', 'up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'up', 'AP'].edge_attr = edge_attr_ue_up_ap
    
    
    data.ap_id = ap_id
    data.sample_id = sample_id

    return data


def create_graph_cen(Beta_all, Gamma_all, Phi_all):
    num_sample, num_AP, num_UE = Beta_all.shape
    data_list = []
    for each_sample in range(num_sample):
        data = full_het_graph(
            Beta_all[each_sample],
            Gamma_all[each_sample],
            None,
            Phi_all[each_sample],
        )
        data_list.append(data)
    return data_list


def build_cen_loader(betaMatrix, gammaMatrix, phiMatrix, batchSize, isShuffle=False):
    log_large_scale = np.log1p(betaMatrix)
    data_cen = create_graph_cen(log_large_scale, gammaMatrix, phiMatrix)
    loader_cen = DataLoader(data_cen, batch_size=batchSize, shuffle=isShuffle)
    return data_cen, loader_cen



# =====================================================================
# Homogeneous graph for downlink cf-mMIMO
# =====================================================================

def full_homo_graph(
        beta_single_sample, gamma_single_sample, 
        label_single_all, phi_single_sample, 
        ap_id=None, sample_id=None, 
        device=None
        
    ):
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_AP, num_UE = beta_single_sample.shape
    num_nodes = num_AP * num_UE 

    # Node index: 0 -> (num_AP * num_UE - 1)
    # node i: ap_idx = i // num_UE -> ue_idx = i % num_UE
    ap_indices = np.repeat(np.arange(num_AP), num_UE)
    ue_indices = np.tile(np.arange(num_UE), num_AP)


    # NODE FEATURE ========================================================
    node_beta = beta_single_sample.flatten()
    node_gamma = gamma_single_sample.flatten()

    if phi_single_sample.ndim == 1:
        node_phi = phi_single_sample[ue_indices].reshape(-1, 1)
    else:
        node_phi = phi_single_sample[ue_indices]

    x_features = np.column_stack((node_beta, node_gamma, node_phi))
    x = torch.tensor(x_features, dtype=torch.float32).to(device)


    # EDGE ATTRRIBUTE =====================================================
    src, dst = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))
    src = src.flatten()
    dst = dst.flatten()

    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long).contiguous().to(device)

    ap_src = src // num_UE
    ue_src = src % num_UE
    
    ap_dst = dst // num_UE
    ue_dst = dst % num_UE

    # Src AP -> Dst UE
    beta_src_to_dst = beta_single_sample[ap_src, ue_dst]
    gamma_src_to_dst = gamma_single_sample[ap_src, ue_dst]
    
    # Dst AP -> Src UE
    beta_dst_to_src = beta_single_sample[ap_dst, ue_src]
    gamma_dst_to_src = gamma_single_sample[ap_dst, ue_src]
    
    edge_features = np.column_stack((
        beta_src_to_dst, gamma_src_to_dst, 
        beta_dst_to_src, gamma_dst_to_src
    ))

    edge_attr = torch.tensor(edge_features, dtype=torch.float32).to(device)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=None)

    data.ap_id = ap_id
    data.sample_id = sample_id
    data.num_AP = num_AP
    data.num_UE = num_UE

    return data


def create_homo_graph(Beta_all, Gamma_all, Phi_all, device):
    num_sample, num_AP, num_UE = Beta_all.shape
    data_list = []
    for each_sample in range(num_sample):
        data = full_homo_graph(
            Beta_all[each_sample],
            Gamma_all[each_sample],
            None,  # label_single_all (None trong trường hợp này)
            Phi_all[each_sample],
            sample_id=each_sample, # Thêm sample_id để dễ tracking nếu cần
            device=device
        )
        data_list.append(data)
        
    return data_list


def build_homo_loader(betaMatrix, gammaMatrix, phiMatrix, batchSize, isShuffle=False, device=None):
    log_large_scale = np.log1p(betaMatrix)
    data_cen = create_homo_graph(log_large_scale, gammaMatrix, phiMatrix, device=device)
    loader_cen = DataLoader(data_cen, batch_size=batchSize, shuffle=isShuffle)
    return data_cen, loader_cen


# =====================================================================
# Sum-rate loss / training / evaluation for centralized baseline
# =====================================================================

def cen_loss_function_sumrate(graphData, nodeFeatDict, edgeDict, tau, rho_p, rho_d, num_antenna, epochRatio=1, eval_mode=False):
    num_graph = graphData.num_graphs
    criterion = nn.MSELoss(reduction='mean') 
    
        
    # label_power = torch.sqrt(graphData.y)
    num_APs = graphData['AP'].x.shape[0]//num_graph
    num_UEs = graphData['UE'].x.shape[0]//num_graph
    
    large_scale = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,0]
    large_scale = torch.expm1(large_scale)
    power_matrix_raw = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,-1]
    # ap_gate = nodeFeatDict['AP'].reshape(num_graph, num_APs, -1)
    phi_matrix = graphData['UE'].x[:,:tau].reshape(num_graph, num_UEs, -1)
    # channel_var = variance_calculate(large_scale, phi_matrix, tau=tau, rho_p=rho_p)
    channel_var = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,1]
    # p_max = (1.0 / num_antenna) ** 0.5
    # den = torch.logsumexp(power_matrix_raw + torch.log(channel_var), dim=2, keepdim=True)
    # term_1 = torch.exp(0.5 * (power_matrix_raw-den))
    # term_2 = torch.sigmoid(torch.sum(power_matrix_raw, dim=2, keepdim=True))
    # term_2 = term_2 ** 0.5
    # power_matrix = p_max  * term_1 * term_2 # Sqrt of power 
    power_matrix = power_from_raw(power_matrix_raw, channel_var, num_antenna)
    
    # rate = rate_calculation(power_matrix, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
    
    all_DS, all_PC, all_UI = component_calculate(power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d)
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d)
    
    if torch.isnan(rate).any():
        print(power_matrix_raw)
        raise ValueError('Nan in rate')
    
    
    if eval_mode:
        sum_rate = torch.sum(rate,dim=1)
        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(full, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
        sum_rate_one = torch.sum(rate_full_one,dim=1)
        return sum_rate, sum_rate_one
    else:
        epochRatio = min(1.0, epochRatio)
        sum_rate = torch.sum(rate,dim=1)
        sum_rate_detach = torch.sum(rate.detach(),dim=1)
        loss = torch.mean(-sum_rate)

        return loss, torch.mean(sum_rate_detach.detach())
    

def cen_train_sumrate( epochRatio,
        dataLoader, model, optimizer,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True) 
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss, _ = cen_loss_function_sumrate(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs



@torch.no_grad()
def cen_eval_sumrate(
        dataLoader, model,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_min_rate = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        _, sum_rate = cen_loss_function_sumrate(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        
        total_min_rate += sum_rate.item() * num_graph
        total_graphs += num_graph

    return total_min_rate/total_graphs 


## Homo graph functions

def train_sumrate_homo( epochRatio,
        dataLoader, model, optimizer,
        tau, rho_p, rho_d, num_antenna, device=None
    ):
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True) 
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x = model(batch)
        loss, _ = loss_function_sumrate_homo(
            batch, x,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs



@torch.no_grad()
def eval_sumrate_homo(
        dataLoader, model,
        tau, rho_p, rho_d, num_antenna, device=None
    ):
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_min_rate = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x = model(batch)
        _, sum_rate = loss_function_sumrate_homo(
            batch, x,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        
        total_min_rate += sum_rate.item() * num_graph
        total_graphs += num_graph

    return total_min_rate/total_graphs 


def loss_function_sumrate_homo(batch, x, tau, rho_p, rho_d, num_antenna, epochRatio=1, eval_mode=False):
    batch_size = batch.num_graphs

    num_AP = batch.num_AP[0].item() if isinstance(batch.num_AP, torch.Tensor) else batch.num_AP[0]
    num_UE = batch.num_UE[0].item() if isinstance(batch.num_UE, torch.Tensor) else batch.num_UE[0]

    beta_flat = batch.x[:, 0]
    gamma_flat = batch.x[:, 1]
    phi_flat = batch.x[:, 2:]
    power_flat = x


    large_scale = beta_flat.view(batch_size, num_AP, num_UE)
    large_scale = torch.expm1(large_scale)
    channel_var = gamma_flat.view(batch_size, num_AP, num_UE)
    power_matrix_raw = power_flat.view(batch_size, num_AP, num_UE)
    
    phi_reshaped = phi_flat.view(batch_size, num_AP, num_UE, tau)
    phi_matrix = phi_reshaped[:, 0, :]  # Shape: [batch_size, num_UE]

    power_matrix = power_from_raw(power_matrix_raw, channel_var, num_antenna)

    all_DS, all_PC, all_UI = component_calculate(power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d)
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d)

    
    if torch.isnan(rate).any():
        print(power_matrix_raw)
        raise ValueError('Nan in rate')
    
    if eval_mode:
        sum_rate = torch.sum(rate,dim=1)
        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(full, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
        sum_rate_one = torch.sum(rate_full_one,dim=1)
        return sum_rate, sum_rate_one
    else:
        epochRatio = min(1.0, epochRatio)
        sum_rate = torch.sum(rate,dim=1)
        sum_rate_detach = torch.sum(rate.detach(),dim=1)
        loss = torch.mean(-sum_rate)

        return loss, torch.mean(sum_rate_detach.detach())