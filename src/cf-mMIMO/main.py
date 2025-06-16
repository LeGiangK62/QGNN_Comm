import os
import torch
import matplotlib.pyplot as plt
import pennylane as qml
import argparse
from torch import nn, optim
import numpy
import scipy.io
from torch_geometric.loader import DataLoader


from utils import train, test, EarlyStopping
from comm_utils import normalize_data, rate_loss
from data import cfGraphDataset, load_cf_dataset

from model import QGNN

from datetime import datetime
import time


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


root_dir = '../..'
result_dir = os.path.join(root_dir, 'results')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, 'fig'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'log'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'model'), exist_ok=True)

param_file = os.path.join(result_dir, 'log', f"{timestamp}_model_parameters.txt")
grad_file = os.path.join(result_dir, 'log', f"{timestamp}_model_gradients.txt")


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN on Wireless Communication System")
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--eval_size', type=int, default=150)
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--node_qubit', type=int, default=3)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_ent_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1712)
    # parser.add_argument('--task', type=str, default='graph', choices=['graph', 'node'], help='graph or node classification')
    
    
    # System parameters
    parser.add_argument('--dataset', type=str, default='d2d', choices=['d2d', 'cf-mmimo'])
    parser.add_argument('--num_ap', type=int, default=30)
    parser.add_argument('--num_ue', type=int, default=6)
    # parser.add_argument('--power', type=float, default=10)
    # parser.add_argument('--noise', type=float, default=10)

    
    # Debug options
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--save_model', action='store_true', help='Enable saving model')
    parser.add_argument('--gradient', action='store_true', help='Enable gradient saving')
    parser.add_argument('--results', action='store_true', help='Evaluate results')
    
    # For switching between models
    parser.add_argument('--model', type=str, default='qgnn', 
                        choices=['qgnn', 'gnn'],
                        help="Which model to run"
                        )
    parser.add_argument('--graphlet_size', type=int, default=10)
    
    
    return parser.parse_args()


def main(args):
    args.node_qubit = args.graphlet_size
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") 
    q_dev = qml.device("default.qubit", wires=n_qubits + 2) # number of ancilla qubits

    # PQC weight shape settings
    w_shapes_dict = {
        'spreadlayer': (0, n_qubits, 1),
        'strong': (2, args.num_ent_layers, 3, 3), # 3
        # 'strong': (3, args.num_ent_layers, 2, 3), # 2
        'inits': (1, 4),
        'update': (1, args.num_ent_layers, 3, 3), # (1, args.num_ent_layers, 2, 3)
        'twodesign': (0, args.num_ent_layers, 1, 2)
    }
    
    result_base = f"{timestamp}_{args.model}_{args.graphlet_size}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}_CF"
    plot_train_path = os.path.join(result_dir, 'fig', f"plot_{result_base}_train.png")
    plot_CDF_path = os.path.join(result_dir, 'fig', f"plot_{result_base}_CDF.png")
    npz_path = os.path.join(result_dir, 'fig', f"data_{result_base}.npz")

    # Load dataset  
    train_path = os.path.join(root_dir, 'data', f'cf_train_{args.num_ue}_{args.num_ap}.mat')
    test_path = os.path.join(root_dir, 'data', f'cf_test_{args.num_ue}_{args.num_ap}.mat')
    norm_train_losses, direct_train, cross_train, (train_K, train_M),\
        norm_test_losses, direct_test, cross_test, (test_K, test_M), \
        opt_rate = load_cf_dataset(train_path, test_path,
            training_sam=args.train_size, testing_sam=args.test_size)
    train_dataset = cfGraphDataset(norm_train_losses, direct_train, cross_train, (train_K, train_M))
    test_dataset = cfGraphDataset(norm_test_losses, direct_test, cross_test, (test_K, test_M))

    # if task_type != 'graph':
    #     raise NotImplementedError("Node classification support is not implemented yet.")
 
    # Model metadata
    node_input_dim = train_dataset[0][0].x.shape[1]
    edge_input_dim = train_dataset[0][0].edge_attr.shape[1]
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'qgnn':
        model = QGNN(
            q_dev=q_dev,
            w_shapes=w_shapes_dict,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            graphlet_size=args.node_qubit,
            hop_neighbor=args.num_gnn_layers,
        )
    elif args.model == 'gnn':
        from baseline import GNN_Cf
        model = GNN_Cf(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            num_layers=args.num_gnn_layers,
            hidden_channels=args.hidden_channels
        )
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented yet.")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
      
    # Training and Tesing preparetion
    training_sinr = []
    testing_sinr = []
    model.train()
    
    model_save = os.path.join(result_dir, 'model', f"{timestamp}_{args.model}_{args.epochs}_{args.lr}_CF.pt")
    early_stopping = EarlyStopping(patience=10, save_path=model_save)
    print(f"{timestamp} \n")
    print(f"Training model with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
          f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")
    step_plot = args.epochs // 10 if args.epochs > 10 else 1
    start = time.time()
    for epoch in range(args.epochs):
        # Train the model
        avg_train_loss = train(model, train_loader, optimizer)
        avg_test_sinr = test(model, test_loader)
        scheduler.step()
        if args.save_model:
            early_stopping(-avg_test_sinr, model)
        training_sinr.append(-avg_train_loss)
        testing_sinr.append(-avg_test_sinr)
        if epoch % step_plot == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, Training loss: {avg_train_loss:.4f}, "
                  f"Training SINR: {training_sinr[-1]:.4f}, Testing SINR: {testing_sinr[-1]:.4f}")
    
    
    end = time.time()
    print(f"Total execution time: {end - start:.6f} seconds")
    
    if args.plot:
        numpy.savez_compressed(
            npz_path, 
            epoch=numpy.arange(1, args.epochs + 1), 
            train_sinr=training_sinr, 
            test_sinr=testing_sinr
        )
        epochs_range = range(1, args.epochs + 1)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, [-x for x in training_sinr], label="Train Loss")
        plt.title(f"Training Loss ({args.graphlet_size}-node Graphlet)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, training_sinr, label="Train SINR")
        plt.plot(epochs_range, testing_sinr, label="Test SINR")
        plt.title(f"SINR vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("SINR (dB)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(plot_train_path, dpi=300)
    
    if args.results:     
        print("Evaluating results...")
        eval_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)

        for data, direct, cross in eval_loader:
            bs = data.num_graphs
            M = direct.shape[1]
            K = data.x.shape[0] // M // bs
            optimizer.zero_grad()

            output = model(data.x, data.edge_attr, data.edge_index, data.batch) # .reshape(bs, -1)
            # output = output.reshape(bs,-1)
            power = output.reshape(bs, M, K)
            power = torch.mean(power, dim=2)
            qgnn_rates = rate_loss(power, direct, cross, True).flatten().detach().numpy()

            full = torch.ones_like(power)
            all_one_rates = rate_loss(full, direct, cross, True).flatten().numpy()
            
        opt_rates = opt_rate
        num_ep = args.test_size
        min_rate, max_rate = 0, 2
        y_axis = numpy.arange(0, 1.0, 1/(num_ep+2))
        qgnn_rates.sort(); all_one_rates.sort(); opt_rates.sort()
        qgnn_rates = numpy.insert(qgnn_rates, 0, min_rate); qgnn_rates = numpy.insert(qgnn_rates,num_ep+1,max_rate)
        all_one_rates = numpy.insert(all_one_rates, 0, min_rate); all_one_rates = numpy.insert(all_one_rates,num_ep+1,max_rate)
        opt_rates = numpy.insert(opt_rates, 0, min_rate); opt_rates = numpy.insert(opt_rates,num_ep+1,max_rate)
        
        plt.figure()
        plt.plot(qgnn_rates, y_axis, label = 'QGNN')
        # plt.plot(gnn_rates, y_axis, label = 'GNN')
        plt.plot(opt_rates, y_axis, label = 'Optimal')
        plt.plot(all_one_rates, y_axis, label = 'Maximum Power')
        plt.xlabel('Minimum rate [bps/Hz]', {'fontsize':16})
        plt.ylabel('Empirical CDF', {'fontsize':16})
        plt.legend(fontsize = 12)
        plt.grid()

        plt.savefig(plot_CDF_path, dpi=300)

    # train_losses = []
    # test_losses = []
    # train_accs = []
    # test_accs = []

    # # Training loop
    # if args.gradient:
    #     string = "="*10 + f"{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}" + "="*10
    #     with open(param_file, "w") as f_param:
    #         f_param.write(string + "\n")
    #     with open(grad_file, "w") as f_grad:
    #         f_grad.write(string + "\n")
        
    # start = time.time()
    # step_plot = args.epochs // 10 if args.epochs > 10 else 1
    
    # model_save = os.path.join(result_dir, 'model', f"{timestamp}_{args.model}_{args.dataset.lower()}.pt")
    # early_stopping = EarlyStopping(patience=10, save_path=model_save)
    
    # if args.task == 'graph':
    #     for epoch in range(1, args.epochs + 1):
    #         train_graph(model, optimizer, train_loader, criterion, device)
    #         train_loss, train_acc, f1_train = test_graph(model, train_loader, criterion, device, num_classes)
    #         test_loss, test_acc, f1_test = test_graph(model, test_loader, criterion, device, num_classes)
    #         scheduler.step()
    #         if args.save_model:
    #             early_stopping(test_loss, model)
    #         train_losses.append(train_loss)
    #         test_losses.append(test_loss)
    #         train_accs.append(train_acc)
    #         test_accs.append(test_acc)
    #         ############
    #         if args.gradient:
    #             # === Write model parameters to file ===
    #             with open(param_file, "a") as f_param:
    #                 f_param.write("="*40 + f" Epoch {epoch} " + "="*40 + "\n")
    #                 for name, param in model.named_parameters():
    #                     f_param.write(f"{name}:\n{param.data.cpu().numpy()}\n\n")

    #             # === Write gradients to separate file ===
    #             with open(grad_file, "a") as f_grad:
    #                 f_grad.write("="*40 + f" Epoch {epoch} " + "="*40 + "\n")
    #                 for name, param in model.named_parameters():
    #                     if param.requires_grad:
    #                         if param.grad is None:
    #                             f_grad.write(f"{name}: No gradient (None)\n")
    #                         else:
    #                             grad = param.grad.cpu().numpy()
    #                             f_grad.write(f"{name}:\n{grad}\n\n")
    #         ############
    #         if epoch % step_plot == 0:
    #             print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
    #                 f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    # else:  # node task
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 
    #         mode='min',                           
    #         factor=args.gamma,      # Multiplies LR by this factor (e.g., 0.5)
    #         patience=args.epochs//10,# Wait this many epochs without improvement
    #         # verbose=True                            # Print updates
    #     )
    #     from utils import train_node, test_node
    #     for epoch in range(1, args.epochs + 1):
    #         train_loss = train_node(model, optimizer, data, criterion, device)
    #         test_metrics = test_node(model, data, criterion, device, num_classes)
    #         train_losses.append(test_metrics['train']['loss'])
    #         test_losses.append(test_metrics['test']['loss'])
    #         train_accs.append(test_metrics['train']['acc'])
    #         test_accs.append(test_metrics['test']['acc'])
    #         scheduler.step(test_metrics['val']['loss'])
    #         if epoch % step_plot == 0:
    #             print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} |" +
    #                 f"Train Acc: {test_metrics['train']['acc']:.4f} | "
    #                 f"Val Acc: {test_metrics['val']['acc']:.4f} | Test Acc: {test_metrics['test']['acc']:.4f}")
    # end = time.time()
    # print(f"Total execution time: {end - start:.6f} seconds")
    # if args.plot:
    #     epochs_range = range(1, args.epochs + 1)

    #     plt.figure(figsize=(10, 5))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(epochs_range, train_losses, label="Train Loss")
    #     plt.plot(epochs_range, test_losses, label="Test Loss")
    #     plt.title("Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.legend()

    #     plt.subplot(1, 2, 2)
    #     plt.plot(epochs_range, train_accs, label="Train Acc")
    #     plt.plot(epochs_range, test_accs, label="Test Acc")
    #     plt.title("Accuracy")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Accuracy")
    #     plt.legend()

    #     plt.tight_layout()
    #     # plot_path = f"plot_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
    #     plot_path = f"plot_{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
    #     plt.savefig(os.path.join('../results/fig', plot_path), dpi=300)
        
    # if args.results:
    #     accuracies = []
    #     num_runs = 100  
        
    #     for each in range(num_runs):
    #         eval_loader = eval_dataset(
    #             name=args.dataset,
    #             path='../data',
    #             eval_size=args.eval_size,
    #             batch_size=args.batch_size,
    #             seed=args.seed+each
    #         )
    #         _, eval_acc, _ = test_graph(model, eval_loader, criterion, device, num_classes)
    #         accuracies.append(eval_acc)

    #     mean_acc = np.mean(accuracies)
    #     std_acc = np.std(accuracies, ddof=1)  # unbiased std deviation

    #     print(f"{args.model} Mean Accuracy: {mean_acc:.4f} ± {std_acc:.3f}")

if __name__ == "__main__":
    args = get_args()
    main(args)
