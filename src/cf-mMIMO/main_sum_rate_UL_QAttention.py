"""
Command-line training script for CF-mMIMO uplink max-sum-rate experiments.

Expected project structure:
    project_root/
        cf_data.py
        train_utils.py
        uplink/
            squid_bipartite_attention.py
            squid_bipartite_Qattention.py
        Data/
            cf_30_6_cvx_wmmse.mat

Examples:
    python train_sumrate_ul_qattention.py --model attention --mat_path Data/cf_30_6_cvx_wmmse.mat --num_samples 50 --epochs 1000
    python train_sumrate_ul_qattention.py --model qattention --mat_path Data/cf_30_6_cvx_wmmse.mat --num_samples 50 --epochs 1000
"""

import argparse
import json
import os
import random
import pickle
from pathlib import Path

import numpy as np
import torch

from cf_data import make_loaders
from train_utils import (
    evaluate,
    compute_eta_baselines,
    train_one_epoch_semi_fixed,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, meta):
    common_kwargs = dict(
        ap_in=meta["ap_dim"],
        ue_in=meta["ue_dim"],
        edge_dim=meta["edge_dim"],
        hidden_dim=args.hidden_dim,
        pqc_dim=args.pqc_dim,
        top_l_aps=args.top_l_aps,
        top_l_ues=args.top_l_ues,
        n_qubits_amp=args.n_qubits_amp,
        q_layers=args.q_layers,
        mp_layers=args.mp_layers,
        q_device=args.q_device,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        noise_std=args.noise_std,
        output_temperature=args.output_temperature,
        output_bias_init=args.output_bias_init,
        use_softmax_power=args.use_softmax_power,
        edge_scale=args.edge_scale,
        epsilon_random=args.epsilon_random,
    )

    if args.model == "attention":
        from uplink.squid_bipartite_attention import BipartiteSQUIDGNNPowerControl_Attention
        return BipartiteSQUIDGNNPowerControl_Attention(**common_kwargs)

    if args.model == "qattention":
        from uplink.squid_bipartite_Qattention import BipartiteSQUIDGNNPowerControl_QAttention
        return BipartiteSQUIDGNNPowerControl_QAttention(**common_kwargs)

    raise ValueError(f"Unknown model: {args.model}. Use 'attention' or 'qattention'.")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Train UL SQUID-GNN/QAttention for CF-mMIMO max-sum-rate.")

    # Data
    parser.add_argument("--mat_path", type=str, default="Data/cf_30_6_cvx_wmmse.mat")
    parser.add_argument("--test_mat", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--use_log_beta", action="store_true")
    parser.add_argument("--no_normalize_beta", action="store_true")

    # Model
    parser.add_argument("--model", type=str, default="qattention", choices=["attention", "qattention"])
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--pqc_dim", type=int, default=2)
    parser.add_argument("--top_l_aps", type=int, default=4)
    parser.add_argument("--top_l_ues", type=int, default=4)
    parser.add_argument("--n_qubits_amp", type=int, default=None)
    parser.add_argument("--q_layers", type=int, default=1)
    parser.add_argument("--mp_layers", type=int, default=3)
    parser.add_argument("--q_device", type=str, default="default.qubit")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=0.8)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--output_temperature", type=float, default=1.0)
    parser.add_argument("--output_bias_init", type=float, default=-1.0)
    parser.add_argument("--use_softmax_power", action="store_true")
    parser.add_argument("--edge_scale", type=float, default=5.0)
    parser.add_argument("--epsilon_random", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=7e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--objective", type=str, default="sum", choices=["sum", "min"])
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--lambda_rate", type=float, default=0.5)
    parser.add_argument("--scheduler_step", type=int, default=50)
    parser.add_argument("--scheduler_gamma", type=float, default=0.8)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0, help="0 disables periodic checkpoints")

    # Runtime/output
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="runs_ul_qattention")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.model}_MSE{args.lambda_mse}_RATE{args.lambda_rate}_LAP{args.top_l_aps}_LUE{args.top_l_ues}_seed{args.seed}"

    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "args.json", vars(args))

    print("=" * 80)
    print("Loading data")
    print("=" * 80)
    train_loader, test_loader, meta = make_loaders(
        train_mat=args.mat_path,
        test_mat=args.test_mat,
        train_ratio=args.train_ratio,
        num_samples=args.num_samples,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        top_l_aps=args.top_l_aps,
        seed=args.seed,
        normalize_beta=not args.no_normalize_beta,
        use_log_beta=args.use_log_beta,
    )
    print("meta:", meta)
    save_json(out_dir / "meta.json", meta)

    print("=" * 80)
    print("Building model")
    print("=" * 80)
    model = build_model(args, meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print("Trainable params:", num_params)

    print("=" * 80)
    print("Computing baselines")
    print("=" * 80)
    cvx_base, wmmse_base = compute_eta_baselines(test_loader, device, args.objective)
    cvx_base_train, wmmse_base_train = compute_eta_baselines(train_loader, device, args.objective)
    print(f"CVX test baseline:   {cvx_base}")
    print(f"WMMSE test baseline: {wmmse_base}")
    print(f"CVX train baseline:  {cvx_base_train}")
    print(f"WMMSE train baseline:{wmmse_base_train}")

    baseline_data = {
        "cvx_base": cvx_base,
        "wmmse_base": wmmse_base,
        "cvx_base_train": cvx_base_train,
        "wmmse_base_train": wmmse_base_train,
    }
    save_json(out_dir / "baselines.json", baseline_data)

    print("=" * 80)
    print("Training")
    print("=" * 80)
    history = []
    best_test = -1e18

    for epoch in range(args.epochs):
        train_info = train_one_epoch_semi_fixed(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            objective=args.objective,
            lambda_mse=args.lambda_mse,
            lambda_rate=args.lambda_rate,
        )

        if (epoch % args.eval_every == 0) or (epoch == args.epochs - 1):
            test_metric = evaluate(model, test_loader, device, args.objective)
        else:
            test_metric = history[-1][4] if history else float("nan")

        row = [
            epoch,
            train_info["loss"],
            train_info["mse"],
            train_info["rate_loss"],
            test_metric,
        ]
        history.append(row)

        train_rate = -train_info["rate_loss"]
        print(
            f"Epoch {epoch:04d} | "
            f"loss={train_info['loss']:.6f} | "
            f"mse={train_info['mse']:.6f} | "
            f"train_rate={train_rate:.6f} | "
            f"test_rate={test_metric:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        if test_metric > best_test:
            best_test = test_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "meta": meta,
                    "best_test": best_test,
                    "baselines": baseline_data,
                    "history": history,
                },
                out_dir / "best.pt",
            )

        if args.save_every and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "meta": meta,
                    "best_test": best_test,
                    "baselines": baseline_data,
                    "history": history,
                },
                out_dir / f"epoch_{epoch+1}.pt",
            )

        scheduler.step()

        # Save history every epoch for safe interruption.
        history_np = np.asarray(history, dtype=np.float64)
        np.savetxt(
            out_dir / "history.csv",
            history_np,
            delimiter=",",
            header="epoch,loss,mse,rate_loss,test_metric",
            comments="",
        )
        with open(out_dir / "history.pkl", "wb") as f:
            pickle.dump(
                {
                    "history": history,
                    "columns": ["epoch", "loss", "mse", "rate_loss", "test_metric"],
                    "baselines": baseline_data,
                    "args": vars(args),
                    "meta": meta,
                },
                f,
            )

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print("Best test metric:", best_test)
    print("Outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
