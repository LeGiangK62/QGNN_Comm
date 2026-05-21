import numpy as numpy
import torch


def normalize_data(train_data,test_data):
    train_mean = numpy.mean(train_data)
    train_std = numpy.std(train_data)
    norm_train = (train_data)/train_std
    norm_test = (test_data)/train_std
    n1, n2 = norm_train.shape[0], norm_test.shape[0]
    return norm_train, norm_test

def rate_loss(allocs, directlink_channel_losses, crosslink_channel_losses, test_mode = False):
    SINRs_numerators = allocs * directlink_channel_losses**2
    SINRs_denominators = torch.squeeze(torch.matmul(crosslink_channel_losses, torch.unsqueeze(allocs, axis=-1))) + directlink_channel_losses
    SINRs = SINRs_numerators / SINRs_denominators
    rates = torch.log2(1 + SINRs)
    min_rate = torch.min(rates, dim = 1)[0] # take min
    if test_mode:
        return min_rate
    else:
        alpha = 0
        loss = - alpha * torch.mean(torch.mean(rates)) - (1 - alpha) * torch.mean(min_rate)
        return loss, torch.mean(min_rate)
    



# Common functions

def variance_calculate(largeScale, phiMatrix, tau, rho_p):
    num = tau * torch.square(largeScale) 

    tmp = torch.square(torch.bmm(
        phiMatrix,
        phiMatrix.transpose(1, 2),
    ).abs())

    largeScale_exp = largeScale.unsqueeze(-1)  # Shape: (num_graphs, num_AP, num_UE, 1)
    tmp_exp = tmp.unsqueeze(1)
    term1 = torch.sum(largeScale_exp * tmp_exp, dim=2)
    denom = tau * term1 + 1

    return num/denom


def rate_calculation(powerMatrix, largeScale, channelVariance, pilotAssignment, rho_d, num_antenna):
    #===========================================
    # sqrt of power 
    # Args:
    # powerMatrix:        power matrix of all AP and UE   [num_samples, num_AP, num_UE]
    # largeScale:         channel large scale fading      [num_samples, num_AP, num_UE]
    # channelVariance:    channel variance                [num_samples, num_AP, num_UE]
    # pilotAssignment:    Pilot Assignment                [num_samples, num_UE, pilot_length]
    # Output
    # rate:               Achievable rate of every UE     [num_samples, num_UE]
    #
    #===========================================    
    # powerMatrix = torch.sqrt(powerMatrix)
    SINR_num = torch.sum(powerMatrix*channelVariance, dim=1) ** 2 * (num_antenna ** 2)
    dtype = powerMatrix.dtype

    powerExpanded = ((powerMatrix**2) *channelVariance).unsqueeze(-1)
    largeScaleExpanded = largeScale.unsqueeze(-2)
    userInterference = torch.sum(powerExpanded * largeScaleExpanded, dim=(1, 2))

    interm_var1 = (powerMatrix * channelVariance/ largeScale).unsqueeze(-1)
    interm_var2 = largeScale.unsqueeze(-2)
    prod = torch.sum(interm_var1 * interm_var2, dim=1) ** 2
    diag_vec = prod.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * torch.eye(powerMatrix.shape[2], device=powerMatrix.device, dtype=dtype)
    
    pilotContamination = torch.bmm(
        pilotAssignment,
        pilotAssignment.transpose(1, 2),
    ).abs()

    pilotContamination = (prod - diag_vec) * pilotContamination
    pilotContamination = torch.sum(pilotContamination, dim=1)


    SINR_denom = 1 + userInterference * num_antenna + pilotContamination * num_antenna ** 2
    rate = torch.log2(1 + SINR_num/SINR_denom)
    return rate

def power_from_raw(rawMatrix, channelVarMatrix, num_antenna=1):
    p_max = (1.0 / num_antenna) ** 0.5
    
    # Option 1
    den = torch.logsumexp(rawMatrix + torch.log(channelVarMatrix), dim=2, keepdim=True)
    term_1 = torch.exp(0.5 * (rawMatrix - den))
    term_2 = torch.sigmoid(torch.sum(rawMatrix, dim=2, keepdim=True))
    term_2 = term_2 ** 0.5
    power_matrix = p_max  * term_1  * term_2 # Sqrt of power 
    
    # Option 2
    # sqrt_p_unnorm = torch.nn.functional.softplus(rawMatrix)
    # power_unnorm = sqrt_p_unnorm**2
    # den = (power_unnorm * channelVarMatrix).sum(dim=2, keepdim=True)
    # scale = torch.clamp(p_max**2 / den, max=1.0)
    # power_matrix = sqrt_p_unnorm * torch.sqrt(scale) 
    
    # Option 3
    # z = torch.softmax(rawMatrix + torch.log(channelVarMatrix), dim=2)
    # term_1 = z / channelVarMatrix
    # term_2 = apRaw # ** 0.5 # No need sqrt, since apRaw in range 0 to 1
    # power_matrix = p_max * (term_1 ** 0.5) * term_2
    
    return power_matrix

# Only FL functions

def rate_from_component(desiredSignal, pilotContamination, userInterference, numAntenna, rho_d=0.1):
    # desiredSignal: num_graphs, num_AP, num_UE
    # pilotContamination: num_graphs, num_AP, num_UE_prime, num_UE
    # userInterference: num_graphs, num_AP, num_UE_prime, num_UE
    
    num_graphs, num_APs, num_UEs = desiredSignal.shape
    device = desiredSignal.device
    dtype = desiredSignal.dtype
    
    sum_DS = desiredSignal.sum(dim=1)  
    num = (numAntenna**2) * (sum_DS ** 2) 

    sum_PC = pilotContamination.sum(dim=1) # num_graphs, num_UE_prime, num_UE
    sum_UI = userInterference.sum(dim=1)  # num_graphs, num_UE_prime, num_UE
    
    sum_PC = sum_PC ** 2
    term1 = sum_PC * (1 - torch.eye(num_UEs, device=device, dtype=dtype))
    term1 = (numAntenna**2) * term1.sum(dim=1)
    # term1 = (numAntenna**2) * ((sum_PC * (1 - torch.eye(num_UEs, device=devcie))).pow(2).sum(dim=1)) 
    term2 = numAntenna * sum_UI.sum(dim=1)          
    
    denom = term1 + term2 + 1

    rate_all = torch.log2(1 + num/denom)  

    return rate_all

def rate_from_component_reduced(desiredSignal, squarePilotContamination, userInterference, numAntenna, rho_d=0.1):
    # desiredSignal: num_graphs, num_AP, num_UE
    # pilotContamination: num_graphs, num_AP, num_UE
    # userInterference: num_graphs, num_AP, num_UE
    num_graphs, num_APs, num_UEs = desiredSignal.shape
    device = desiredSignal.device
    dtype = desiredSignal.dtype
    
    sum_DS = desiredSignal.sum(dim=1)  
    num = (numAntenna**2) * (sum_DS ** 2) 

    # Only sum over APs (dim=1), since already summed over num_UE_prime
    sum_UI = userInterference.sum(dim=1)    # shape: (num_graphs, num_UE)
    
    term1 = (numAntenna**2) * squarePilotContamination  # This is the second sum that was there before
    
    term2 = numAntenna * sum_UI  # This is the second sum that was there before
    
    print(term1.shape)
    print(term2.shape)
    denom = term1 + term2 + 1

    rate_all = torch.log2(1 + num/denom)  

    return rate_all
    

def component_calculate(power, channelVariance, largeScale, phiMatrix, rho_d=0.1):
    #################
    # sqrt of power 
    # power                 : torch.rand(num_graphs, num_AP, num_UE)
    # channelVariance       : torch.rand(num_graphs, num_AP, num_UE)
    # largeScale            : torch.rand(num_graphs, num_AP, num_UE)
    # phiMatrix             : torch.rand(num_graphs, num_UE, tau)
    # out
    # DS: num_graphs, num_AP, num_UE
    # PC: num_graphs, num_AP, num_UE_prime, num_UE
    # UI: num_graphs, num_AP, num_UE_prime, num_UE
    #################
    device = power.device
    
    pilotContamination = torch.bmm(
        phiMatrix,
        phiMatrix.transpose(1, 2),
    ).abs()
    
    DS_all = power * channelVariance 

    tmp = (power**2) * channelVariance
    tmp = tmp.unsqueeze(-1)
    largeScale_expand = largeScale.unsqueeze(-2)
    UI_all = tmp * largeScale_expand

    # mask = torch.eye(UI_all.size(-1), device=device).bool()
    # UI_all[:, :, mask] = 0

    tmp = power * channelVariance / largeScale
    tmp = tmp.unsqueeze(-1)

    tmp = tmp * largeScale_expand
    PC_all = tmp * pilotContamination.unsqueeze(-3)
    # mask = torch.eye(PC_all.size(-1), device=device).bool()
    # PC_all[:, :, mask] = 0


    return DS_all, PC_all, UI_all


# def package_calculate(batch, x_dict, tau, rho_p, rho_d):
#     num_graphs = batch.num_graphs
#     num_UEs = x_dict['UE'].shape[0] // num_graphs
#     num_APs = x_dict['AP'].shape[0] // num_graphs
#     ue_feature = x_dict['UE'].reshape(num_graphs, num_UEs, -1)
#     power = ue_feature[:,:, -1][:,None,:]
#     phiMatrix = ue_feature[:,:, :-1]

#     largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)

#     channelVariance = variance_calculate(largeScale, phiMatrix, tau, rho_p)

    
#     return component_calculate(power, channelVariance, largeScale, phiMatrix, rho=rho_d)