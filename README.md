# QGNN_Comm
Implementation of QGNN on Wireless Communication System


Testing Command 

```bash
python main.py --train_size 200 --test_size 50 --epochs 50 --batch_size 32 --hidden_channels 64 --graphlet_size 5 --num_ent_layers 2 --num_gnn_layers 2 --seed 1712 --lr 0.01 --step_size 5 --gamma 0.8 --num_ap 30 --num_ue 6  --plot --gradient --save_model --results```


# D2D
```bash
python main.py --train_size 300 --test_size 100 --epochs 100 --batch_size 32 --hidden_channels 64 --graphlet_size 3 --num_ent_layers 2 --num_gnn_layers 2 --seed 1712 --lr 0.001 --step_size 10 --gamma 0.8 --num_nodes 20 --power 1 --noise 10  --model qgnn --plot --save_model 
```
Current best - plot_20260319-163602_qgnn_3_100_0.001_D2D_train
