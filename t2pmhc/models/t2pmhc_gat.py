import os
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import copy

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import random
from collections import Counter

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, BatchNorm, AttentionalAggregation, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import wandb


from utils.helpers import save_last_model, get_device






# ============================================================================= #
#                               set seed                                        #
# ============================================================================= #

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


def create_graph_dataset(saved_graphs):
    if os.path.exists(saved_graphs):
        print("Loading Graphs from pt file")
        dataset = torch.load(saved_graphs, weights_only=False)
    else:
        print("Error: Saved graphs file does not exist. Please ensure the file path is correct or set 'load_graphs' to False to generate graphs.")

    return dataset, len(dataset)


def scale_features(train_subset, val_subset):
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
    Returns:
        tuple: (
            train_subset_copy (list): Scaled copy of the training subset,
            val_subset_copy (list): Scaled copy of the validation subset,
            pae_node_scaler (MinMaxScaler): Scaler fitted on node "PAE" values,
            paetcrpmhc_node_scaler (MinMaxScaler): Scaler fitted on node "PAE_TCRpMHC" values,
            distance_scaler (MinMaxScaler): Scaler fitted on edge distance values,
            pae_edge_scaler (MinMaxScaler): Scaler fitted on edge PAE values
        )
    """
    
    # Deep copy to avoid modifying input objects
    train_subset_copy = [copy.deepcopy(graph) for graph in train_subset]
    val_subset_copy = [copy.deepcopy(graph) for graph in val_subset]

    ###### Node feature scaler ######

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
    paes = all_edge_features[:, 1]
    # fit scaler
    distance_scaler = MinMaxScaler().fit(distances.reshape(-1, 1))
    pae_edge_scaler = MinMaxScaler().fit(paes.reshape(-1, 1))


    # Scale PAE values for train and val subsets and add as feature to each graph
    for subset in [train_subset_copy, val_subset_copy]:
        for graph in subset:
            # scale node features
            scale_nodefeatures(pae_node_scaler, paetcrpmhc_node_scaler, hydro_scaler, graph)
            # scale edge features
            scale_edgefeatures(distance_scaler, pae_edge_scaler, graph)

    return train_subset_copy, val_subset_copy, pae_node_scaler, paetcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler
    

def scale_nodefeatures(pae_scaler, paetcrpmhc_scaler, hydro_scaler, graph):
    """
    Scales the PAE, PAE_TCRpMHC and hydrophobicity node-level features for a graph and appends them as new columns to the node feature matrix.
    Args:
        pae_scaler: A fitted scaler object (e.g., from sklearn) for the PAE feature.
        paetcrpmhc_scaler: A fitted scaler object for the PAE_TCRpMHC feature.
        graph: A graph object with a 'meta' dictionary containing 'PAE' and 'PAE_TCRpMHC', and a 'x' attribute for node features.
    Returns:
        None. The function updates the 'x' attribute of the input graph in-place with the new scaled features.
    """
    # get values
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
    # no need to repeat for hydro as each aa has different value
    hydro_feat = torch.tensor(scaled_hydro)

    graph.x = torch.cat([graph.x, pae_feat, paetcrpmhc_feat, hydro_feat], dim=1)


def scale_edgefeatures(distance_scaler, pae_scaler, graph):
    """
    Scales the edge features of a graph using provided scalers for distance and PAE (Predicted Aligned Error).
    This function extracts the distance and PAE features from the graph's edge features, applies the corresponding
    scalers to normalize or standardize these values, and then combines the scaled features into a new edge attribute
    tensor for the graph.
    Args:
        distance_scaler: A fitted scaler object (e.g., from sklearn) used to transform the distance feature.
        pae_scaler: A fitted scaler object used to transform the PAE feature.
        graph: A graph object with an 'edge_features' attribute (NumPy array) and an 'edge_attr' attribute to store
            the scaled features as a PyTorch tensor.
    Returns:
        None. The function updates the 'edge_attr' attribute of the input graph in-place with the scaled edge features.
    """

    edge_features = graph.edge_features
    distances = edge_features[:, 0]
    paes = edge_features[:, 1]
    scaled_distances = distance_scaler.transform(distances.reshape(-1, 1))
    scaled_pae = pae_scaler.transform(paes.reshape(-1, 1))
    # Add as new feature (column) to node features
    scaled_features = np.hstack([scaled_distances, scaled_pae]).astype(np.float32)  # Combine distances and PAE values
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
            print("edge index assumption is wrong!")
            sys.exit()

        return node_emb, (edge_index_out, avg_alpha), batch
        

# 
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




def train_gat(metadata_path, name, hyperparams, saved_graphs, save_model):
    print("Training t2pmhc-GAT")

    print(f"\nName: {name}\nSaved Graphs: {saved_graphs}\n")
    print("............. reading dataset ............")

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
    print(f"Training on {device}")


    # Add PAE features across the full dataset
    train_subset_scaled, val_subset_scaled, pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler = scale_features(dataset, [])

    # Create data loader for full dataset
    train_loader = DataLoader(train_subset_scaled, batch_size=batch_size, shuffle=True)

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

    print("Training t2pmhc-GAT model")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")


    # save model
    if not os.path.exists(save_model):
        os.makedirs(save_model)
        
    save_last_model(model, save_model, name)
    print("Final model trained and saved.")
