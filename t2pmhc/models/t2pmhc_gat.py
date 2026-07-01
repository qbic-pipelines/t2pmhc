import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import copy
import logging
import random
from collections import Counter

import torch
from torch_geometric.nn import GATConv, BatchNorm, AttentionalAggregation
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from t2pmhc.utils.helpers import create_peptide_folds


from t2pmhc.utils.helpers import save_last_model, save_last_scalers, get_device



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
    Loads a pre-saved graph dataset from a specified file path.
    If the file exists, it loads the dataset using torch.load and returns the dataset along with its length.
    If the file does not exist, it raises a FileNotFoundError with an appropriate message.
    Args:
        saved_graphs (str): The file path to the saved graph dataset.
    Returns:
        tuple: (dataset, length of dataset)
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if os.path.exists(saved_graphs):
        logger.info("Loading Graphs from pt file")
        dataset = torch.load(saved_graphs, weights_only=False)
    else:
        raise FileNotFoundError("Error: Saved graphs file does not exist. Please ensure the file path is correct or set 'load_graphs' to False to generate graphs.")

    return dataset, len(dataset)


def scale_features(train_subset, val_subset, ablate_pae=False):
    """
    Scales node and edge features for graph datasets using MinMaxScaler.
    This function takes training and validation subsets of graphs, deep copies them to avoid modifying the originals,
    and fits MinMaxScalers on the training data for both node and edge features. Specifically, it scales:
      - Node features: "PAE" and "PAE_TCRpMHC" values from each graph's meta information.
      - Edge features: The first two columns of each graph's edge_features array, assumed to represent distances and PAE values.
    The fitted scalers are then used to transform both the training and validation subsets. The scaled features are added
    back to each graph using the helper functions `scale_nodefeatures` and `scale_edgefeatures`.
    Args:
        train_subset (list): List of graph objects for training, each with 'meta' and 'edge_features' attributes.
        val_subset (list): List of graph objects for validation, each with 'meta' and 'edge_features' attributes.
        ablate_pae (bool): If True, skip all PAE-related features (node PAE, PAE_TCRpMHC, edge PAE).
    Returns:
        tuple: (
            train_subset_copy (list): Scaled copy of the training subset,
            val_subset_copy (list): Scaled copy of the validation subset,
            pae_node_scaler (MinMaxScaler or None): Scaler fitted on node "PAE" values,
            paetcrpmhc_node_scaler (MinMaxScaler or None): Scaler fitted on node "PAE_TCRpMHC" values,
            distance_scaler (MinMaxScaler): Scaler fitted on edge distance values,
            pae_edge_scaler (MinMaxScaler or None): Scaler fitted on edge PAE values,
            hydro_scaler (MinMaxScaler): Scaler fitted on hydrophobicity values
        )
    """

    # Deep copy to avoid modifying input objects
    train_subset_copy = [copy.deepcopy(graph) for graph in train_subset]
    val_subset_copy = [copy.deepcopy(graph) for graph in val_subset]

    ###### Node feature scaler ######

    pae_node_scaler = None
    paetcrpmhc_node_scaler = None
    if not ablate_pae:
        # get training PAEs from the meta object
        pae_vals_train = np.array([graph.meta["PAE"] for graph in train_subset_copy], dtype=np.float32)
        paetcrpmhc_vals_train = np.array([graph.meta["PAE_TCRpMHC"] for graph in train_subset_copy], dtype=np.float32)
        # fit scaler
        pae_node_scaler = MinMaxScaler().fit(pae_vals_train.reshape(-1, 1))
        paetcrpmhc_node_scaler = MinMaxScaler().fit(paetcrpmhc_vals_train.reshape(-1, 1))

    # scale hydrophobicity feature
    hydro_train = np.vstack([graph.meta["hydro"] for graph in train_subset_copy]).astype(np.float32)
    # fit scaler
    hydro_scaler = MinMaxScaler().fit(hydro_train)

    ###### edge features scaler ######

    # get edge features
    all_edge_features = np.concatenate([graph.edge_features for graph in train_subset_copy], dtype=np.float32)
    distances = all_edge_features[:, 0]
    # fit scaler
    distance_scaler = MinMaxScaler().fit(distances.reshape(-1, 1))

    pae_edge_scaler = None
    if not ablate_pae:
        paes = all_edge_features[:, 1]
        pae_edge_scaler = MinMaxScaler().fit(paes.reshape(-1, 1))


    # Scale values for train and val subsets and add as feature to each graph
    for subset in [train_subset_copy, val_subset_copy]:
        for graph in subset:
            # scale node features
            scale_nodefeatures(pae_node_scaler, paetcrpmhc_node_scaler, hydro_scaler, graph, ablate_pae)
            # scale edge features
            scale_edgefeatures(distance_scaler, pae_edge_scaler, graph, ablate_pae)

    return train_subset_copy, val_subset_copy, pae_node_scaler, paetcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler
    

def scale_nodefeatures(pae_scaler, paetcrpmhc_scaler, hydro_scaler, graph, ablate_pae=False):
    """
    Scales the PAE, PAE_TCRpMHC and hydrophobicity node-level features for a graph and appends them as new columns to the node feature matrix.
    Args:
        pae_scaler: A fitted scaler object (e.g., from sklearn) for the PAE feature. None if ablate_pae=True.
        paetcrpmhc_scaler: A fitted scaler object for the PAE_TCRpMHC feature. None if ablate_pae=True.
        hydro_scaler: A fitted scaler object for the hydrophobicity feature.
        graph: A graph object with a 'meta' dictionary containing 'PAE' and 'PAE_TCRpMHC', and a 'x' attribute for node features.
        ablate_pae (bool): If True, skip PAE and PAE_TCRpMHC features.
    Returns:
        None. The function updates the 'x' attribute of the input graph in-place with the new scaled features.
    """
    hydro_vals = graph.meta["hydro"]
    scaled_hydro = hydro_scaler.transform(hydro_vals)
    hydro_feat = torch.tensor(scaled_hydro)

    if not ablate_pae:
        # get values
        pae_val = np.array([[graph.meta["PAE"]]], dtype=np.float32)
        paetcrpmhc_val = np.array([[graph.meta["PAE_TCRpMHC"]]], dtype=np.float32)
        # scale
        scaled_pae = pae_scaler.transform(pae_val)
        scaled_paetcrpmhc = paetcrpmhc_scaler.transform(paetcrpmhc_val)
        # Add as new feature (column) to node features
        pae_feat = torch.tensor(scaled_pae, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
        paetcrpmhc_feat = torch.tensor(scaled_paetcrpmhc, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
        graph.x = torch.cat([graph.x, pae_feat, paetcrpmhc_feat, hydro_feat], dim=1)
    else:
        graph.x = torch.cat([graph.x, hydro_feat], dim=1)


def scale_edgefeatures(distance_scaler, pae_scaler, graph, ablate_pae=False):
    """
    Scales the edge features of a graph using provided scalers for distance and PAE (Predicted Aligned Error).
    Args:
        distance_scaler: A fitted scaler object (e.g., from sklearn) used to transform the distance feature.
        pae_scaler: A fitted scaler object used to transform the PAE feature. None if ablate_pae=True.
        graph: A graph object with an 'edge_features' attribute (NumPy array) and an 'edge_attr' attribute to store
            the scaled features as a PyTorch tensor.
        ablate_pae (bool): If True, only use distance (no PAE edge feature).
    Returns:
        None. The function updates the 'edge_attr' attribute of the input graph in-place with the scaled edge features.
    """

    edge_features = graph.edge_features
    distances = edge_features[:, 0]
    scaled_distances = distance_scaler.transform(distances.reshape(-1, 1))

    if not ablate_pae:
        paes = edge_features[:, 1]
        scaled_pae = pae_scaler.transform(paes.reshape(-1, 1))
        scaled_features = np.hstack([scaled_distances, scaled_pae]).astype(np.float32)
    else:
        scaled_features = scaled_distances.astype(np.float32)

    graph.edge_attr = torch.tensor(scaled_features, dtype=torch.float)



# ============================================================================= #
#                                t2pmhc-GAT                                     #
# ============================================================================= #

class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, edge_dim, heads):
        super(GATClassifier, self).__init__()
        
        # GAT layers with edge_dim specified
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
                
        # Batch normalization
        self.fn1 = BatchNorm(hidden_dim * heads)
        self.fn2 = BatchNorm(hidden_dim)
        self.fn3 = BatchNorm(hidden_dim)
        
        # Attention-based global pooling
        self.att_pool = AttentionalAggregation(gate_nn=torch.nn.Linear(hidden_dim, 1))
        
        # Final fully connected classifier
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # GAT Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.fn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # GAT Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.fn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # GAT Layer 3
        x = self.conv3(x, edge_index, edge_attr)
        x = self.fn3(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Attention-based pooling
        x = self.att_pool(x, batch)

        # Final classifier
        x = self.fc(x)
        return x
    
    def get_attention_weights(self, data):
        """
        Returns node embeddings + edge attention coefficients for inspection.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Layer 1
        x1, (edge_index1, alpha1) = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        x1 = self.fn1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)

        # Layer 2
        x2, (edge_index2, alpha2) = self.conv2(x1, edge_index, edge_attr, return_attention_weights=True)
        x2 = self.fn2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)

        # Layer 3
        x3, (edge_index3, alpha3) = self.conv3(x2, edge_index, edge_attr, return_attention_weights=True)
        x3 = self.fn3(x3)
        x3 = F.elu(x3)
        x3 = self.dropout(x3)

        # collect node embeddings
        node_emb = x3

        # attn1/2/3 each contain (edge_index, α_ij)
        # average across layers
        alphas = [alpha1, alpha2, alpha3]
        avg_alpha = torch.stack([a.mean(dim=1) if a.dim() > 1 else a for a in alphas]).mean(dim=0)

        # Use the last layer’s edge_index (all layers have same edge structure after self-loops)
        if torch.equal(edge_index3, edge_index3) and torch.equal(edge_index2, edge_index3):
            edge_index_out = edge_index3
        else:
            logger.info("edge index assumption is wrong!")
            sys.exit()

        return node_emb, (edge_index_out, avg_alpha), batch
        

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




def train_gat(metadata_path, name, hyperparams, saved_graphs, save_model, resume_from=None):
    """
    Trains a t2pmhc-GAT model using the provided dataset and hyperparameters.
    Args:
        metadata_path (str): Path to the metadata file.
        name (str): Name identifier for the model.
        hyperparams (dict): Dictionary containing hyperparameters for training.
        saved_graphs (str): Path to the saved graph dataset.
        save_model (str): Directory path to save the trained model.
    """
    logger.info("Training t2pmhc-GAT")

    logger.info(f"\nName: {name}\nSaved Graphs: {saved_graphs}\n")
    logger.info("Reading dataset")

    # set seed
    seed = 42
    set_seed(seed)

    metadata = pd.read_csv(metadata_path, sep="\t")
    dataset, structure_count = create_graph_dataset(saved_graphs)

    # Hyperparameters
    input_dim = hyperparams["input_dim"]
    hidden_dim = hyperparams["hidden_dim"]
    output_dim = hyperparams["output_dim"]
    learning_rate = hyperparams["learning_rate"]
    num_epochs = hyperparams["num_epochs"]
    weight_decay = hyperparams["weight_decay"]
    dropout_rate = hyperparams["dropout_rate"]
    batch_size = hyperparams["batch_size"]
    k = hyperparams["k"]
    edge_dim = hyperparams["edge_dim"]
    heads = hyperparams["heads"]

    # enable GPU usage
    device = get_device()
    logger.info(f"Training on {device}")


    # Add PAE features across the full dataset
    dataset_scaled, _, pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler = scale_features(dataset, [])

    # set reproducible generator
    g = torch.Generator()
    g.manual_seed(seed)

    # Create data loader for full dataset
    train_loader = DataLoader(dataset_scaled,
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=4,
                              persistent_workers=True, 
                              worker_init_fn=seed_worker, 
                              generator=g)

    # Class weights for imbalance
    labels = [data.y.item() for data in dataset]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = torch.tensor([total / counts[c] for c in sorted(counts)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    # Model and optimizer
    model = GATClassifier(input_dim, hidden_dim, output_dim, dropout_rate, edge_dim, heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    # Ensure checkpoint dir exists
    checkpoint_dir = os.path.join(save_model, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Training t2pmhc-GAT model")
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}')

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, f"{name}_latest.pt"))

    # save model
    os.makedirs(save_model, exist_ok=True)

    save_last_model(model, save_model, name)
    save_last_scalers(pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler, name, "GAT", save_model)
    logger.info("Final model trained and saved.")


def train_gat_cv(metadata_path, name, hyperparams, saved_graphs, save_model, ablate_pae=False):
    """
    Train t2pmhc-GAT model with 5-fold stratified cross-validation.
    Supports epoch-level checkpoint/resume and PAE ablation.
    Args:
        metadata_path (str): Path to metadata file.
        name (str): Name for the model.
        hyperparams (dict): Hyperparameters for training.
        saved_graphs (str): Path to saved graphs .pt file.
        save_model (str): Directory to save the trained model.
        ablate_pae (bool): If True, remove all PAE features.
    """
    logger.info("Training t2pmhc-GAT (5-fold CV)")
    logger.info(f"\nName: {name}\nSaved Graphs: {saved_graphs}\nAblate PAE: {ablate_pae}\n")
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
    k = hyperparams["k"]
    edge_dim = hyperparams["edge_dim"]
    heads = hyperparams["heads"]

    device = get_device()
    logger.info(f"Training on {device}")

    labels = np.array([data.y.item() for data in dataset])
    identifiers = [data.meta["id"] for data in dataset]

    if not os.path.exists(save_model):
        os.makedirs(save_model)

    checkpoint_dir = os.path.join(save_model, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    split_records = []

    auroc_path = os.path.join(save_model, f"{name}_cv_auroc.csv")
    splits_path = os.path.join(save_model, f"{name}_cv_splits.tsv")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        logger.info(f"---- Fold {fold}/5 ----")

        fold_name = f"{name}_fold{fold}"
        model_path = os.path.join(save_model, f"{fold_name}.pt")
        checkpoint_path = os.path.join(checkpoint_dir, f"{fold_name}_checkpoint.pt")

        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]

        # Always re-fit scalers from the train split (deterministic given fixed seed)
        train_scaled, val_scaled, pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler = \
            scale_features(train_subset, val_subset, ablate_pae=ablate_pae)

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

        model = GATClassifier(input_dim, hidden_dim, output_dim, dropout_rate, edge_dim, heads).to(device)

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

            # Resume from checkpoint if it exists
            start_epoch = 0
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                logger.info(f"Fold {fold}: resuming from epoch {start_epoch}")

            for epoch in range(start_epoch, num_epochs):
                train_loss = train(model, train_loader, optimizer, criterion, device)
                logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Fold {fold} | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}')

                # Save checkpoint after each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

            save_last_model(model, save_model, fold_name)
            save_last_scalers(pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler, fold_name, "GAT", save_model)

            # Clean up checkpoint after successful fold completion
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

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


def train_gat_peptide_cv(metadata_path, name, hyperparams, saved_graphs, save_model, ablate_pae=False):
    """
    Train t2pmhc-GAT model with 5-fold peptide-grouped cross-validation.
    All samples sharing the same peptide are assigned to the same fold.
    Supports epoch-level checkpoint/resume and PAE ablation.
    Args:
        metadata_path (str): Path to metadata file.
        name (str): Name for the model.
        hyperparams (dict): Hyperparameters for training.
        saved_graphs (str): Path to saved graphs .pt file.
        save_model (str): Directory to save the trained model.
        ablate_pae (bool): If True, remove all PAE features.
    """
    logger.info("Training t2pmhc-GAT (5-fold peptide-grouped CV)")
    logger.info(f"\nName: {name}\nSaved Graphs: {saved_graphs}\nAblate PAE: {ablate_pae}\n")
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
    k = hyperparams["k"]
    edge_dim = hyperparams["edge_dim"]
    heads = hyperparams["heads"]

    device = get_device()
    logger.info(f"Training on {device}")

    labels = np.array([data.y.item() for data in dataset])
    identifiers = [data.meta["id"] for data in dataset]

    if not os.path.exists(save_model):
        os.makedirs(save_model)

    checkpoint_dir = os.path.join(save_model, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create peptide-grouped folds
    fold_assignments, peptide_fold_map, peptide_counts = create_peptide_folds(dataset, n_splits=5)

    # Log and save peptide fold assignments
    peptide_folds_path = os.path.join(save_model, f"{name}_peptide_folds.tsv")
    peptide_fold_records = []
    for peptide, fold in sorted(peptide_fold_map.items(), key=lambda x: x[1]):
        peptide_fold_records.append({"peptide": peptide, "fold": fold, "n_samples": peptide_counts[peptide]})
    pd.DataFrame(peptide_fold_records).to_csv(peptide_folds_path, sep="\t", index=False)

    for fold_num in range(1, 6):
        fold_total = sum(peptide_counts[p] for p, f in peptide_fold_map.items() if f == fold_num)
        fold_peptides = [p for p, f in peptide_fold_map.items() if f == fold_num]
        logger.info(f"Fold {fold_num}: {fold_total} samples, {len(fold_peptides)} peptides")

    fold_results = []
    split_records = []

    auroc_path = os.path.join(save_model, f"{name}_cv_auroc.csv")
    splits_path = os.path.join(save_model, f"{name}_cv_splits.tsv")

    for fold in range(1, 6):
        logger.info(f"---- Fold {fold}/5 ----")

        val_idx = np.array(fold_assignments[fold])
        train_idx = np.array([i for f in range(1, 6) if f != fold for i in fold_assignments[f]])

        fold_name = f"{name}_fold{fold}"
        model_path = os.path.join(save_model, f"{fold_name}.pt")
        checkpoint_path = os.path.join(checkpoint_dir, f"{fold_name}_checkpoint.pt")

        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]

        # Always re-fit scalers from the train split
        train_scaled, val_scaled, pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler = \
            scale_features(train_subset, val_subset, ablate_pae=ablate_pae)

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

        model = GATClassifier(input_dim, hidden_dim, output_dim, dropout_rate, edge_dim, heads).to(device)

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

            # Resume from checkpoint if it exists
            start_epoch = 0
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                logger.info(f"Fold {fold}: resuming from epoch {start_epoch}")

            for epoch in range(start_epoch, num_epochs):
                train_loss = train(model, train_loader, optimizer, criterion, device)
                logger.info(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Fold {fold} | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}')

                # Save checkpoint after each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

            save_last_model(model, save_model, fold_name)
            save_last_scalers(pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler, fold_name, "GAT", save_model)

            # Clean up checkpoint after successful fold completion
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

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
    logger.info(f"5-fold peptide-grouped CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    logger.info(f"Saved CV AUROC results to {auroc_path}")
    logger.info(f"Saved split samplesheet to {splits_path}")
    logger.info(f"Saved peptide fold assignments to {peptide_folds_path}")

    logger.info("5-fold peptide-grouped cross-validation complete.")
