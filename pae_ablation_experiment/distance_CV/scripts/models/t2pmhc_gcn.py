import os
import pandas as pd
import numpy as np
import random
import logging
from collections import Counter
from datetime import datetime
import copy


import torch
from torch_geometric.nn import GCNConv, BatchNorm, AttentionalAggregation
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from t2pmhc.utils.helpers import save_last_model, save_last_scalers, get_device




# get logger
logger = logging.getLogger("t2pmhc")


# ============================================================================= #
#                               set seed                                        #
# ============================================================================= #


def set_seed(seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    Args:
        seed (int): The seed value to set for random number generation.
    """
    logger.info(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)




# ============================================================================= #
#                          Structure Representation                             #
# ============================================================================= #


def create_graph_dataset(saved_graphs):
    """
    Load precomputed graph dataset from a .pt file.  
    Args:
        saved_graphs (str): Path to the saved graphs .pt file.
    Returns:
        dataset (list): List of graph objects.
        structure_count (int): Number of structures in the dataset. 
    """
    if os.path.exists(saved_graphs):
        logger.info("Loading Graphs from pt file")
        dataset = torch.load(saved_graphs, weights_only=False)
    else:
        raise FileNotFoundError("Error: Saved graphs file does not exist. Please ensure the file path is correct or set 'load_graphs' to False to generate graphs.")

    return dataset, len(dataset)


def scale_features(train_subset, val_subset):
    """
    Scale PAE, hydro, and distance features using MinMaxScaler.
    Returns scaled train and val subsets along with the fitted scalers.
    Args:
        train_subset (list): List of training graph objects.
        val_subset (list): List of validation graph objects.
        metadata (pd.DataFrame): Metadata DataFrame containing PAE and hydro values.
    Returns:
        train_subset_copy (list): Scaled training graph objects.
        val_subset_copy (list): Scaled validation graph objects.
        pae_scaler (MinMaxScaler): Fitted scaler for PAE values.
        paetcrpmhc_scaler (MinMaxScaler): Fitted scaler for PAE_TCRpMHC values.
        hydro_scaler (MinMaxScaler): Fitted scaler for hydro values.
        distance_scaler (MinMaxScaler): Fitted scaler for edge distances.
    """
    # Deep copy to avoid modifying input objects
    train_subset_copy = [copy.deepcopy(graph) for graph in train_subset]
    val_subset_copy = [copy.deepcopy(graph) for graph in val_subset]

    # get training PAEs from the meta object
    pae_vals_train = np.array([graph.meta["PAE"] for graph in train_subset_copy], dtype=np.float32)
    paetcrpmhc_vals_train = np.array([graph.meta["PAE_TCRpMHC"] for graph in train_subset_copy], dtype=np.float32)

    # fit scaler
    pae_scaler = MinMaxScaler().fit(pae_vals_train.reshape(-1, 1))
    paetcrpmhc_scaler = MinMaxScaler().fit(paetcrpmhc_vals_train.reshape(-1, 1))

    # scale hydro
    hydro_train = np.vstack([graph.meta["hydro"] for graph in train_subset_copy]).astype(np.float32)
    # fit scaler
    hydro_scaler = MinMaxScaler().fit(hydro_train)

    # distance scaler (edge)
    all_edge_features = np.concatenate([graph.edge_features for graph in train_subset_copy], dtype=np.float32)
    distances = all_edge_features[:,0]
    #fit scaler
    distance_scaler = MinMaxScaler().fit(distances.reshape(-1,1))

    # Scale PAE values for train and val subsets and add as feature to each graph
    for subset in [train_subset_copy, val_subset_copy]:
        for graph in subset:
            # read values
            pae_val = np.array([[graph.meta["PAE"]]], dtype=np.float32)
            paetcrpmhc_val = np.array([[graph.meta["PAE_TCRpMHC"]]], dtype=np.float32)
            hydro_vals = graph.meta["hydro"]
            # scale
            scaled_pae = pae_scaler.transform(pae_val)
            scaled_paetcrpmhc = paetcrpmhc_scaler.transform(paetcrpmhc_val)
            scaled_hydro = hydro_scaler.transform(hydro_vals)
            # Add as new feature (column) to node features
            pae_feat = torch.tensor(scaled_pae, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
            paetcrpmhc_feat = torch.tensor(scaled_paetcrpmhc, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
            # hydro each aa has different value
            hydro_feat = torch.tensor(scaled_hydro)
            graph.x = torch.cat([graph.x, pae_feat, paetcrpmhc_feat, hydro_feat], dim=1)

            # scale edge features
            edge_features = graph.edge_features
            distances = edge_features[:,0]
            scaled_distances = distance_scaler.transform(distances.reshape(-1,1))
            graph.edge_attr = torch.tensor(scaled_distances, dtype=torch.float)

    return train_subset_copy, val_subset_copy, pae_scaler, paetcrpmhc_scaler, hydro_scaler, distance_scaler


# ============================================================================= #
#                                t2pmhc-GCN                                     #
# ============================================================================= #


class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GCNClassifier, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
                
        # Batch normalization
        self.fn1 = BatchNorm(hidden_dim)
        self.fn2 = BatchNorm(hidden_dim)
        self.fn3 = BatchNorm(hidden_dim)
        
        # Residual connection projection
        self.residual_fc = torch.nn.Linear(input_dim, hidden_dim)
        
        # Attention-based global pooling
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Linear(hidden_dim, 1))
        
        # Fully connected classifier
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
            
        # GCN Layer 1 with residual
        x = self.conv1(x, edge_index)
        x = self.fn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN Layer 2
        x = self.conv2(x, edge_index)
        x = self.fn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN Layer 3
        x = self.conv3(x, edge_index)
        x = self.fn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Attention-based pooling
        x = self.att_pool(x, batch)
        
        # Final classifier
        x = self.fc(x)

        return x
    
    def get_attention_weights(self, data):
        """
        Return node embeddings and attention weights per node for each graph.
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Forward through GCN layers (same as in forward())
        x = self.conv1(x, edge_index)
        x = self.fn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.fn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.fn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Compute attention logits
        gate_logits = self.att_pool.gate_nn(x).squeeze()  # [num_nodes]

        # Subtract max per graph for numerical stability
        unique_batches = batch.unique()
        max_per_graph = torch.zeros_like(gate_logits)
        for b in unique_batches:
            mask = (batch == b)
            max_per_graph[mask] = gate_logits[mask].max()

        stabilized = gate_logits - max_per_graph

        # exp
        exp_x = stabilized.exp()

        # sum per group
        sum_per_graph = torch.zeros_like(exp_x)
        for b in unique_batches:
            mask = (batch == b)
            sum_per_graph[mask] = exp_x[mask].sum()

        attn_weights = exp_x / (sum_per_graph + 1e-16)

        return x, attn_weights, batch
        

# Training function
def train(model, loader, optimizer, criterion, device):
    # put the model in train mode
    model.train()
    # running total to accumulate loss over all batches
    total_loss = 0
    for data in loader:
        # move to gpu if possible
        data = data.to(device)
        # clears old gradients from previous step
        optimizer.zero_grad()
        # forward pass
        out = model(data)
        # loss
        loss = criterion(out, data.y)
        # backpropagation
        loss.backward()
        # update model weights
        optimizer.step()
        total_loss += loss.item()
    # return average loss per batch for monitoring
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, return_probs=False):
    # put model in evaluation mode
    model.eval()
    # init statistics
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    all_preds = []

    # disble gradient tracking (performance and memory efficiency)
    with torch.no_grad():
        for data in loader:
            # move to gpu if possible
            data = data.to(device)
            # forward pass
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            # 2-class classification
            probs = F.softmax(out, dim=1)[:, 1]  # Probability of class 1 (binder)
            # give out predicted class
            pred = out.argmax(dim=1)

            # update stats
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    if return_probs:
        return avg_loss, accuracy, all_labels, all_probs, all_preds
    return avg_loss, accuracy


def train_gcn(metadata_path, name, hyperparams, saved_graphs, save_model):
    """
    Train t2pmhc-GCN model with 5-fold stratified cross-validation.
    Saves one model and set of scalers per fold, a split samplesheet, and per-fold AUROC results.
    Args:
        metadata_path (str): Path to metadata file.
        name (str): Name for the model.
        hyperparams (dict): Hyperparameters for training.
        saved_graphs (str): Path to saved graphs .pt file.
        save_model (str): Directory to save the trained model.
    """
    logger.info("Training t2pmhc-GCN")
    logger.info(f"\nName: {name}\nSaved Graphs: {saved_graphs}\n")
    logger.info("Reading dataset")

    seed = 42
    set_seed(seed)

    metadata = pd.read_csv(metadata_path, sep="\t")

    dataset, structure_count = create_graph_dataset(saved_graphs)
    logger.info(f"Loaded {structure_count} graphs")

    # Hyperparameters
    input_dim = hyperparams["input_dim"]
    hidden_dim = hyperparams["hidden_dim"]
    output_dim = hyperparams["output_dim"]
    learning_rate = hyperparams["learning_rate"]
    num_epochs = hyperparams["num_epochs"]
    weight_decay = hyperparams["weight_decay"]
    dropout_rate = hyperparams["dropout_rate"]
    batch_size = hyperparams["batch_size"]

    device = get_device()
    logger.info(f"Training on {device}")

    labels = np.array([data.y.item() for data in dataset])
    identifiers = [data.meta["id"] for data in dataset]

    if not os.path.exists(save_model):
        os.makedirs(save_model)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    split_records = []

    auroc_path = os.path.join(save_model, f"{name}_cv_auroc.csv")
    splits_path = os.path.join(save_model, f"{name}_cv_splits.tsv")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        logger.info(f"---- Fold {fold}/5 ----")

        fold_name = f"{name}_fold{fold}"
        model_path = os.path.join(save_model, f"{fold_name}.pt")

        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]

        # Always re-fit scalers from the train split (deterministic given fixed seed)
        train_scaled, val_scaled, pae_scaler, pae_tcrpmhc_scaler, hydro_scaler, distance_scaler = \
            scale_features(train_subset, val_subset)

        val_loader = DataLoader(val_scaled,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                persistent_workers=True)

        train_labels = labels[train_idx]
        counts = Counter(train_labels.tolist())
        total = sum(counts.values())
        class_weights = torch.tensor([total / counts.get(c, total) for c in [0, 1]], dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        model = GCNClassifier(input_dim, hidden_dim, output_dim, dropout_rate).to(device)

        if os.path.exists(model_path):
            logger.info(f"Fold {fold}: model already exists, skipping training and evaluating saved model")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_scaled,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=4,
                                      persistent_workers=True,
                                      worker_init_fn=seed_worker,
                                      generator=g)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, optimizer, criterion, device)
                logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Fold {fold} | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}')

            save_last_model(model, save_model, fold_name)
            save_last_scalers(pae_scaler, pae_tcrpmhc_scaler, distance_scaler, "", hydro_scaler, fold_name, "GCN", save_model)
            del train_loader, optimizer

        _, _, val_labels_list, val_probs, _ = evaluate(model, val_loader, criterion, device, return_probs=True)
        auroc = roc_auc_score(val_labels_list, val_probs)
        logger.info(f"Fold {fold} AUROC: {auroc:.4f}")
        fold_results.append({"run_name": name, "fold": fold, "auroc": auroc, "n_train": len(train_idx), "n_val": len(val_idx)})

        for i in train_idx:
            split_records.append({"identifier": identifiers[i], "label": labels[i], "fold": fold, "split": "train"})
        for i in val_idx:
            split_records.append({"identifier": identifiers[i], "label": labels[i], "fold": fold, "split": "val"})

        # Save results after each fold so partial runs produce usable output
        pd.DataFrame(fold_results).to_csv(auroc_path, index=False)
        split_df = pd.DataFrame(split_records)
        split_df[split_df["split"] == "val"][["identifier", "label", "fold"]].rename(columns={"fold": "val_fold"}).to_csv(splits_path, sep="\t", index=False)

        # Free memory before next fold
        del model, criterion, val_loader, train_scaled, val_scaled, train_subset, val_subset
        torch.cuda.empty_cache()

    # Log final summary
    results_df = pd.DataFrame(fold_results)
    mean_auroc = results_df["auroc"].mean()
    std_auroc = results_df["auroc"].std()
    logger.info(f"5-fold CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    logger.info(f"Saved CV AUROC results to {auroc_path}")
    logger.info(f"Saved split samplesheet to {splits_path}")

    logger.info("5-fold cross-validation complete.")