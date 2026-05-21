'''
This file training homogeneous QGNN (batched) for downlink sum-rate cf-mMIMO
xxx


python main_sumrate_DL.py --train_size 500 --test_size 200 --eval_size 500 --graphlet_size 5 --plot --results

python main_sumrate_DL.py --num_train 500 --num_test 200 --num_eval 500 --graphlet_size 5
'''


import time
import os
import argparse

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import scipy.io
import torch
import numpy as np
import pennylane as qml

from cf_utils import (
    build_homo_loader,
    train_sumrate_homo,
    eval_sumrate_homo,
    loss_function_sumrate_homo,
    # build_cen_loader,
    # cen_train_sumrate,
    # cen_eval_sumrate,
    # cen_loss_function_sumrate,
)

from baseline import HomoCfmMimoNet

from cf_model import QGNN_DL as QGNN

root_dir = '../..'
SAVE_DIR = os.path.join(root_dir, 'results', 'cf_sumrate')
MODEL_DIR = SAVE_DIR + "/models/"
EVAL_DIR = SAVE_DIR + "/eval/"
FIG_DIR = SAVE_DIR + "/figs"
TRAIN_DIR = SAVE_DIR + "/train/"


def init_folder():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN on cf-mMIMO Sum-rate Maximization")

    # General / IO
    parser.add_argument('--seed', type=int, default=1712, help="Random seed")
    parser.add_argument('--qgnn_pretrain', type=str, default=None,
                        help="Name of QGNN model to load directly without training")
    parser.add_argument('--c_pretrain', type=str, default=None,
                        help="Name of centralized GNN model to load directly without training")
    parser.add_argument('--eval_plot', action='store_true', default=True,
                        help="Eval Visualization (CDF)")

    # System (cf-mMIMO communication) parameters
    parser.add_argument('--num_ap', type=int, default=30, help="Number of access points")
    parser.add_argument('--num_ue', type=int, default=6, help="Number of user equipments")
    parser.add_argument('--tau', type=int, default=20, help="Pilot length")
    parser.add_argument('--power_f', type=float, default=0.2, help="Transmit power threshold")
    parser.add_argument('--D', type=float, default=1, help="Area diameters (km)")
    parser.add_argument('--num_antenna', type=int, default=1, help="Number of antennas per AP")

    # Data
    parser.add_argument('--num_train', type=int, default=500, help="Number of training samples")
    parser.add_argument('--num_test', type=int, default=200, help="Number of testing samples")
    parser.add_argument('--num_eval', type=int, default=200, help="Number of evaluation samples")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")

    # QGNN hyperparameters (main model)
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Number of QGNN training epochs")
    parser.add_argument('--lr', type=float, default=5e-3, help="QGNN learning rate")
    parser.add_argument('--lr_c', type=float, default=5e-3,
                        help="Learning rate for classical params of QGNN")
    parser.add_argument('--lr_q', type=float, default=5e-3,
                        help="Learning rate for quantum params of QGNN")
    parser.add_argument('--gamma', type=float, default=0.8, help="StepLR decay factor")
    parser.add_argument('--step_size', type=int, default=5, help="StepLR step size (epochs)")
    parser.add_argument('--hidden_channels', type=int, default=32,
                        help="Number of hidden channels for QGNN classical layers")
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help="Number of QGNN hops/message-passing layers")
    parser.add_argument('--graphlet_size', type=int, default=10,
                        help="Star-graphlet size (== node qubits)")
    parser.add_argument('--node_qubit', type=int, default=3,
                        help="(Overridden by graphlet_size) Number of node qubits")
    parser.add_argument('--num_ent_layers', type=int, default=1,
                        help="Number of entangling layers per PQC")
    parser.add_argument('--q_dev', type=str, default="default.qubit",
                        help="PennyLane quantum device")

    # Centralized GNN (benchmark) hyperparameters
    parser.add_argument('--cen_lr', type=float, default=5e-3, help="Centralized GNN learning rate")
    parser.add_argument('--num_epochs_cen', type=int, default=50,
                        help="Number of centralized training epochs")
    parser.add_argument('--cen_hidden_channels', type=int, default=32,
                        help="Number of hidden channels for centralized GNN")
    parser.add_argument('--cen_num_gnn_layers', type=int, default=3,
                        help="Number of centralized GNN layers")

    return parser.parse_args()


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")

    timestamp = time.strftime('%y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    init_folder()

    args = get_args()
    seed = args.seed

    # Training param
    qgnn_pretrain = args.qgnn_pretrain
    c_pretrain = args.c_pretrain

    # QGNN hyperparams
    hidden_channels = args.hidden_channels
    num_gnn_layers = args.num_gnn_layers
    num_epochs = args.num_epochs
    lr = args.lr

    # Centralized hyperparams
    num_epochs_cen = args.num_epochs_cen
    cen_lr = args.cen_lr

    # Training meta
    num_train = args.num_train
    num_test = args.num_test
    num_eval = args.num_eval
    batch_size = args.batch_size

    # Communication params
    tau = args.tau
    num_antenna = args.num_antenna
    rho_p, rho_d = args.power_f, args.power_f
    num_ap = args.num_ap
    num_ue = args.num_ue

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------------------- Load data ----------------------
    file_name = f'dl_sumrate_data_2000_{num_ue}_{num_ap}'
    mat_data = scipy.io.loadmat('Data/' + file_name + '.mat')
    beta_all = mat_data['betas']
    gamma_all = mat_data['Gammas']
    phi_all = mat_data['Phii_cf'].transpose(0, 2, 1)
    rates_equal_solutions = mat_data['R_equal'][0]
    rates_frac_solutions = mat_data['R_frac'][0]
    rates_log_solutions = mat_data['R_log'][0]

    perm = np.random.RandomState(seed).permutation(beta_all.shape[0])
    train_idx = perm[:num_train]
    test_idx = perm[num_train: num_train + num_test]
    eval_idx = perm[-num_eval:]

    # Centralized loaders (shared by QGNN and centralized GNN benchmark)
    train_data, train_loader = build_homo_loader(
        beta_all[train_idx], gamma_all[train_idx], phi_all[train_idx],
        batch_size, isShuffle=True, device=device
    )
    test_data, test_loader = build_homo_loader(
        beta_all[test_idx], gamma_all[test_idx], phi_all[test_idx],
        batch_size, device=device
    )
    eval_data, eval_loader = build_homo_loader(
        beta_all[eval_idx], gamma_all[eval_idx], phi_all[eval_idx],
        num_eval, device=device
    )

    # Model meta
    node_dim = train_data[0].x.shape[1]
    edge_dim = train_data[0].edge_attr.shape[1]

    # ---------------------- QGNN (main model) ----------------------

    aux_qubit = 1 
    args.node_qubit = args.graphlet_size
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    device = torch.device("cpu") 
    q_dev = qml.device("default.qubit", wires=n_qubits + aux_qubit) # number of ancilla qubits
    print(f'Quantum device: {n_qubits} qubit - {q_dev}')

    w_shapes_dict = {
        'inits': (args.num_ent_layers, 4), 
        'strong': (args.num_ent_layers, 4), 
        'update': (edge_qubit, args.num_ent_layers, 2 + aux_qubit, 3), 
    }

    # # PQC setup
    # args.node_qubit = args.graphlet_size
    # edge_qubit = args.node_qubit - 1
    # n_qubits = args.node_qubit + edge_qubit
    # n_auxi = 0
    # upd_qubits = 2 + n_auxi
    # q_dev = qml.device(args.q_dev, wires=n_qubits + n_auxi)

    # w_shapes_dict = {
    #     'spreadlayer': (0, n_qubits, 1),
    #     'inits': (1, 6),
    #     'strong': (2, args.num_ent_layers, 2, 3),
    #     'update': (args.graphlet_size, args.num_ent_layers - 1, upd_qubits, 3),
    #     'twodesign': (0, args.num_ent_layers, 1, 2),
    # }

    # node_input_dim = {'UE': ue_dim, 'AP': ap_dim}
    # edge_input_dim = {'UE': edge_dim, 'AP': edge_dim}

    # # NOTE: QGNN placeholder. The user will update model.py:QGNN so its
    # # forward signature matches the centralized sum-rate training pipeline
    # # (i.e. takes a HeteroData `batch` and returns (x_dict, edge_dict, edge_index)).
    qgnn_model = QGNN(
        q_dev=q_dev,
        w_shapes=w_shapes_dict,
        hidden_dim=hidden_channels,
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        graphlet_size=args.node_qubit,
        hop_neighbor=num_gnn_layers,
    ).to(device)

    quantum_params, classical_params = [], []
    for name, param in qgnn_model.named_parameters():
        if "qconvs" in name:
            quantum_params.append(param)
        else:
            classical_params.append(param)

    qgnn_optimizer = torch.optim.Adam([
        {"params": classical_params, "lr": args.lr_c},
        {"params": quantum_params, "lr": args.lr_q},
    ])
    qgnn_scheduler = torch.optim.lr_scheduler.StepLR(
        qgnn_optimizer, step_size=args.step_size, gamma=args.gamma
    )

    qgnn_all_rate = []
    qgnn_all_rate_test = []


    if qgnn_pretrain is not None:
        qgnn_model_filename = f'{MODEL_DIR}/{qgnn_pretrain}.pth'
        qgnn_model.load_state_dict(torch.load(qgnn_model_filename))
        print(f'Loaded pre-trained QGNN from {qgnn_model_filename}.')
    else:
        start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n ===={start_time}==== Training QGNN on {device} ... ")
        start_time = time.time()
        eval_epochs = num_epochs // 10 if num_epochs // 10 else 1
        print(f'Training QGNN ({args.graphlet_size}-node graphlet, '
              f'{num_gnn_layers} hops, lr={lr}, epochs={num_epochs})')
        print(f'Equal Power rate: train {np.mean(rates_equal_solutions[train_idx]):.4f}, '
              f'test {np.mean(rates_equal_solutions[test_idx]):.4f}')
        print(f'Log Approximation rate: train {np.mean(rates_log_solutions[train_idx]):.4f}, '
              f'test {np.mean(rates_log_solutions[test_idx]):.4f}')

        for epoch in range(num_epochs):
            qgnn_model.train()
            train_loss = train_sumrate_homo(
                epoch / (2 * num_epochs // 3),
                train_loader, qgnn_model, qgnn_optimizer,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
                device=device
            )

            qgnn_model.eval()
            with torch.no_grad():
                train_eval = eval_sumrate_homo(
                    train_loader, qgnn_model,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna, 
                    device=device
                )
                test_eval = eval_sumrate_homo(
                    test_loader, qgnn_model,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna, 
                    device=device
                )
            qgnn_all_rate.append(train_eval)
            qgnn_all_rate_test.append(test_eval)
            qgnn_scheduler.step()

            if epoch % eval_epochs == 0:
                print(
                    f"Epoch {epoch+1:03d}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Rate: {train_eval:.4f} | "
                    f"Test Rate: {test_eval:.4f}"
                )

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {timedelta(seconds=execution_time)}")

        plt.figure(figsize=(6, 4), dpi=180)
        plt.plot(qgnn_all_rate, label='Training Rate', linewidth=2)
        plt.plot(qgnn_all_rate_test, label='Testing Rate', linewidth=2)
        plt.axhline(y=np.mean(rates_log_solutions[train_idx]), linewidth=2,
                    color='r', linestyle='--', label='Training Log Approx.')
        plt.axhline(y=np.mean(rates_log_solutions[test_idx]), linewidth=2,
                    color='b', linestyle='--', label='Testing Log Approx.')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title(f'QGNN Training Rate Curve - lr {lr}, {num_epochs} epochs', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        save_path = TRAIN_DIR + f'{timestamp}_qgnn.png'
        plt.savefig(save_path, dpi=300)

        qgnn_model_filename = f'{MODEL_DIR}/{timestamp}_qgnn.pth'
        torch.save(qgnn_model.state_dict(), qgnn_model_filename)
        print(f'Save QGNN to {qgnn_model_filename}.')

    # ---------------------- Centralized GNN (benchmark) ----------------------
    c_model = HomoCfmMimoNet(
        node_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_channels=args.cen_hidden_channels,
        num_layers=args.cen_num_gnn_layers // 2,
    ).to(device)
    torch.nn.utils.clip_grad_norm_(c_model.parameters(), 1.0)
    c_optimizer = torch.optim.AdamW(
        c_model.parameters(), lr=cen_lr, weight_decay=1e-4
    )
    c_scheduler = torch.optim.lr_scheduler.StepLR(
        c_optimizer, step_size=num_epochs_cen // 10, gamma=0.8
    )

    if c_pretrain is not None:
        model_filename = f'{MODEL_DIR}/{c_pretrain}.pth'
        c_model.load_state_dict(torch.load(model_filename))
    else:
        start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n ===={start_time}==== Training Centralized GNN (benchmark) ... ")
        start_time = time.time()
        all_rate = []
        all_rate_test = []
        eval_epochs_cen = num_epochs_cen // 10 if num_epochs_cen // 10 else 1
        print(f'Optimal rate: train {np.mean(rates_log_solutions[train_idx]):.4f}, '
              f'test {np.mean(rates_log_solutions[test_idx]):.4f}')
        for epoch in range(num_epochs_cen):
            c_model.train()

            train_loss = train_sumrate_homo(
                epoch / (2 * num_epochs_cen // 3),
                train_loader, c_model, c_optimizer,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
            )

            c_model.eval()
            with torch.no_grad():
                train_eval = eval_sumrate_homo(
                    train_loader, c_model,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
                )
                test_eval = eval_sumrate_homo(
                    test_loader, c_model,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
                )
            all_rate.append(train_eval)
            all_rate_test.append(test_eval)
            c_scheduler.step()
            if epoch % eval_epochs_cen == 0:
                print(
                    f"Epoch {epoch+1:03d}/{num_epochs_cen} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Rate: {train_eval:.4f} | "
                    f"Test Rate: {test_eval:.4f}"
                )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {timedelta(seconds=execution_time)}")

        plt.figure(figsize=(6, 4), dpi=180)
        plt.plot(all_rate, label='Training Rate', linewidth=2)
        plt.plot(all_rate_test, label='Testing Rate', linewidth=2)
        plt.axhline(y=np.mean(rates_log_solutions[train_idx]), linewidth=2,
                    color='r', linestyle='--', label='Training Log Approx.')
        plt.axhline(y=np.mean(rates_log_solutions[test_idx]), linewidth=2,
                    color='b', linestyle='--', label='Testing Log Approx.')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title('Centralized GNN Training Rate Curve', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        save_path = TRAIN_DIR + f'{timestamp}_cen.png'
        plt.savefig(save_path, dpi=300)

        model_filename = f'{MODEL_DIR}/{timestamp}_cen.pth'
        torch.save(c_model.state_dict(), model_filename)
        print(f'Save centralized GNN to {model_filename}.')

    # ---------------------- Evaluation - CDF ----------------------
    if args.eval_plot:
        print('Evaluation' + '=' * 20)

        c_model.eval()
        # qgnn_model.eval()

        for batch in eval_loader:
            batch = batch.to(device)

            # Centralized GNN benchmark
            x = c_model(batch)
            gnn_rates, all_one_rates = loss_function_sumrate_homo(
                batch, x,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
                eval_mode=True
            )

            # # QGNN (placeholder: assumes same interface as cen_model after user updates)
            # q_x_dict, q_edge_dict, q_edge_index = qgnn_model(batch)
            # qgnn_rates, _ = loss_function_sumrate_homo(
            #     batch, q_x_dict, q_edge_dict,
            #     tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            #     eval_mode=True
            # )

        # qgnn_rates = qgnn_rates.detach().cpu().numpy()

        gnn_rates = gnn_rates.detach().cpu().numpy()
        all_one_rates = all_one_rates.detach().cpu().numpy()
        rates_equal = rates_equal_solutions[eval_idx]
        rates_frac = rates_frac_solutions[eval_idx]
        rates_log = rates_log_solutions[eval_idx]

        qgnn_rates = all_one_rates

        max_value = np.ceil(
            max(np.max(all_one_rates), np.max(qgnn_rates), np.max(gnn_rates),
                np.max(rates_equal), np.max(rates_frac), np.max(rates_log)) * 100
        ) / 100

        print(f'Sum rate avg: GNN {gnn_rates.mean():.2f} - '
              f'QGNN {qgnn_rates.mean():.2f} - '
              f'{qgnn_rates.mean() * 100 / gnn_rates.mean():.2f}%')

        min_rate, max_rate = 0, max_value
        y_axis = np.linspace(0, 1, num_eval + 2)
        gnn_rates.sort(); all_one_rates.sort(); qgnn_rates.sort()
        rates_equal.sort(); rates_frac.sort(); rates_log.sort()
        gnn_rates = np.insert(gnn_rates, 0, min_rate)
        gnn_rates = np.insert(gnn_rates, num_eval + 1, max_rate)
        qgnn_rates = np.insert(qgnn_rates, 0, min_rate)
        qgnn_rates = np.insert(qgnn_rates, num_eval + 1, max_rate)
        all_one_rates = np.insert(all_one_rates, 0, min_rate)
        all_one_rates = np.insert(all_one_rates, num_eval + 1, max_rate)
        rates_equal = np.insert(rates_equal, 0, min_rate)
        rates_equal = np.insert(rates_equal, num_eval + 1, max_rate)
        rates_frac = np.insert(rates_frac, 0, min_rate)
        rates_frac = np.insert(rates_frac, num_eval + 1, max_rate)
        rates_log = np.insert(rates_log, 0, min_rate)
        rates_log = np.insert(rates_log, num_eval + 1, max_rate)

        plt.figure(figsize=(6, 4), dpi=180)
        plt.plot(gnn_rates, y_axis, label='Centralized GNN', linewidth=2)
        # plt.plot(qgnn_rates, y_axis, label='QGNN', linewidth=2)
        plt.plot(rates_equal, y_axis, label='Equal Power', linewidth=2)
        plt.plot(rates_log, y_axis, label='Log Approx.', linewidth=2)
        plt.xlabel('Sum rate [bps/Hz]', {'fontsize': 16})
        plt.ylabel('Empirical CDF', {'fontsize': 16})
        plt.legend(fontsize=14)
        plt.grid()

        eval_path = EVAL_DIR + f'/{timestamp}_eval.png'
        plt.savefig(eval_path, dpi=300, bbox_inches='tight')
        print(f'Save Evaluation figure to {eval_path}.')
