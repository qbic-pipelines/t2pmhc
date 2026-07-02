import numpy as np
import argparse
from collections import Counter
from datetime import datetime
import gc

import torch
from torch_geometric.nn import GATConv, BatchNorm, AttentionalAggregation, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import wandb

from t2pmhc.models.t2pmhc_gat import train as gat_train, create_graph_dataset, evaluate as gat_evaluate, scale_features
from t2pmhc.utils.helpers import get_device


class FlexibleGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, edge_dim, heads, num_layers):
        super(FlexibleGAT, self).__init__()
        assert num_layers >= 1, "Number of layers must be at least 1"
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim))
        self.bns.append(BatchNorm(hidden_dim * heads))

        # Second layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Third layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim))
        self.bns.append(BatchNorm(hidden_dim))


        self.dropout = nn.Dropout(dropout_rate)

        # Attention-based global pooling
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Linear(hidden_dim, 1))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        x = self.att_pool(x, batch)
        x = self.fc(x)
        return x


def hyperparam_search(dataset, wandb_config, device):

    # use stratifiedKFold
    labels = [data.y.item() for data in dataset]

    all_auc = []
    
    config = wandb_config

    skf = StratifiedKFold(n_splits=config.k, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n................ Cross-Val Fold {fold + 1}/{config.k} ................")
        print(f"  Fold {fold + 1} -> Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        # ADD MEMORY CLEANUP AT START OF EACH FOLD
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
            
        try:
            # get train/val subsets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # add pae feature and scale features
            train_subset_scaled, val_subset_scaled, pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler = scale_features(train_subset, val_subset)


            train_loader = DataLoader(train_subset_scaled, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset_scaled, batch_size=config.batch_size, shuffle=False)
        except Exception as E:
            print(f"Error in fold {fold+1} setupt: {e}")
            print("Skipping this fold")
            continue

        train_labels = [dataset[i].y.item() for i in train_idx]
        counts = Counter(train_labels)
        total = sum(counts.values())
        class_weights = torch.tensor([total / counts[c] for c in sorted(counts)], dtype=torch.float)
        class_weights = class_weights / class_weights.sum()

        model = FlexibleGAT(
            input_dim=33,
            hidden_dim=config.hidden_dim,
            output_dim=2,
            dropout_rate=config.dropout,
            edge_dim=2,
            heads=config.num_heads,
            num_layers=config.num_layers
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        # early stopping
        patience = 5
        best_auc_so_far = 0
        no_improve_count = 0


        print("................ training ................")
        for epoch in range(config.num_epochs):
            try:
                train_loss = gat_train(model, train_loader, optimizer, criterion, device)
                wandb.log({'loss' : train_loss, 'epoch' : epoch})


                # Evaluation
                val_loss, val_accuracy, val_labels, val_probs, val_preds = gat_evaluate(model, val_loader, criterion, device=device, return_probs=True)

                print(f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # calc tpr, fpr
                fpr, tpr, _ = roc_curve(val_labels, val_probs)
                # calc auc
                auc_score = roc_auc_score(val_labels, val_probs)
                auc01_score = roc_auc_score(val_labels, val_probs, max_fpr=0.1)
                
                wandb.log({'val_loss' : val_loss,
                        'auc' : auc_score, 
                        'auc01' : auc01_score,
                        "roc_curve": wandb.plot.line_series(
                                xs=fpr,
                                ys=[tpr],
                                keys=["Fold ROC"],
                                title="ROC Curve",
                                xname="False Positive Rate"
                            )
                        })
                
                print(f"[Fold {fold+1}] AUC: {auc_score:.4f}, AUC@0.1: {auc01_score:.4f}, time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Early stopping logic starting at epoch 15
                if epoch > 14:
                    if auc_score > best_auc_so_far + 1e-4:
                        best_auc_so_far = auc_score
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch} (AUC hasn't improved for {patience} epochs).")
                        break
            except Exception as e:
                print(f"Error in epoch {epoch} of fold {fold+1}: {e}")
                print("Continuing to next epoch...")
                continue

        # log confusion matrix
        try:
            cm = confusion_matrix(val_labels, val_preds)
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
                y_true=val_labels, preds=val_preds, class_names=["Neg", "Pos"]
            )})

            # append last auc of the fold to all_auc
            all_auc.append(auc_score)
        except Exception as e:
            print(f"Error logging confusion matrix for fold {fold+1}: {e}")
            # Still append AUC if we have it
            if 'auc_score' in locals():
                all_auc.append(auc_score)
        
        # CLEANUP AT END OF FOLD
        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # ADD CHECK FOR EMPTY all_auc
    if len(all_auc) == 0:
        print("No successful folds completed!")
        wandb.log({"error": "No successful folds"})
        wandb.finish()
        return 0.0

    # Log boxplot of all_auc to wandb
    #wandb.log({"auc_boxplot": wandb.plot.box(all_auc, title="AUC Distribution Across Folds")})
    
    # Log as summary statistics
    wandb.log({
    "all_auc_mean": np.mean(all_auc),
    "all_auc_std": np.std(all_auc),
    "all_auc_min": np.min(all_auc),
    "all_auc_max": np.max(all_auc),
    "all_auc_values": all_auc
    })

    wandb.finish()

    return np.mean(all_auc)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_sweep', type=str, default=None,
                       help='Sweep ID to resume (if provided)')
    parser.add_argument('--max_runs', type=int, default=None,
                       help='Maximum number of runs for this job')
    parser.add_argument('--graphs', type=str, default=None,
                        help='Path to t2pmhc graphs')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name to log to')
    args = parser.parse_args()

    print("Hyperparam search for GAT")


    # wandb tracking stuff
    # set sweep config
    sweep_config = {
        'method' : 'bayes',
    }
    # set metric
    metric = {
        'name' : 'auc',
        'goal' : 'maximize',

    }
    # add early terminate
    early_terminate = {
        'type': 'hyperband',
        'min_iter': 10
    }
    # add stuff to config
    sweep_config["metric"] = metric
    sweep_config["early_terminate"] = early_terminate

    # set param dict
    parameters_dict = {
        'hidden_dim': {
            'distribution': 'q_uniform',
            'q': 32,
            'min': 64,
            'max': 128
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.005
        },
        'num_layers': {'values': [2, 3, 4, 5]},  # Keep discrete for small ranges
        'num_heads': {'values': [4, 6, 8]}, 
        'batch_size': {'value': 8},  # Keep discrete
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 0.001
        },
        'input_dim': {'value': 33},
        'output_dim': {'value': 2},
        'k': {'value': 5},
        'num_epochs': {'values': [50,100,150]},
    }
    sweep_config["parameters"] = parameters_dict


    # init sweep
    if args.resume_sweep:
        print(f"Resuming sweep: {args.resume_sweep}")
        sweep_id = args.resume_sweep
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        print(f"Created new sweep: {sweep_id}")

    print("............. reading dataset ............")
    # read in graphs
    graphs = args.graphs

    # Load graphs
    dataset, structure_count = create_graph_dataset(graphs)

    # train
    def sweep_train():
        with wandb.init(notes="GAT sweep"):
            wandb.config.update({"architecture": "GAT", "activation": "ELU"})
            config = wandb.config
            hyperparam_search(dataset, config, device)

    # check if possible to run on gpu
    device = get_device()
    print(f"Training on {device}")
    print("starting sweep")
    wandb.agent(sweep_id, sweep_train, count=args.max_runs)





if __name__ == "__main__":
    main()