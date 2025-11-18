import os
import tqdm
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import random
from collections import Counter
import json
import copy

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import gc

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm, AttentionalAggregation, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import torch_scatter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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




# ============================================================================= #
#                          Structure Representation                             #
# ============================================================================= #



def create_graph_dataset(saved_graphs):
    if os.path.exists(saved_graphs):
        print("Loading Graphs from pt file")
        dataset = torch.load(saved_graphs, weights_only=False)
    else:
        print("Error: Saved graphs file does not exist. Please ensure the file path is correct or set 'load_graphs' to False to generate graphs.")

    return dataset, len(dataset)


def scale_features(train_subset, val_subset, metadata):

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
        """Return node embeddings and attention weights per node for each graph."""
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

        # Compute attention logits and softmax per graph (for benchmarking)
        gate_logits = self.att_pool.gate_nn(x).squeeze()  # [num_nodes]
        attn_weights = torch_scatter.composite.scatter_softmax(gate_logits, batch)

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



def train_gcn(metadata_path, name, hyperparams, saved_graphs, save_model):
    print("Training t2pmhc-GCN")

    print(f"\nName: {name}\nSaved Graphs: {saved_graphs}\n")
    print("............. reading dataset ............")

    metadata = pd.read_csv(metadata_path, sep="\t")

    # read in graphs
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

    # check device
    device = get_device()
    print(f"Training on {device}")

    # get labels
    labels = [data.y.item() for data in dataset]

    # Add PAE features across the full dataset (scale features)
    dataset_scaled, _, pae_scaler, pae_tcrpmhc_scaler, hydro_scaler, distance_scaler = scale_features(dataset, [], metadata)

    # Create data loader for full dataset
    train_loader = DataLoader(dataset_scaled, batch_size=batch_size, shuffle=True)

    # Class weights for imbalance
    labels = [data.y.item() for data in dataset]
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = torch.tensor([total / counts[c] for c in sorted(counts)], dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    # Model and optimizer
    model = GCNClassifier(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("Training t2pmhc-GCN model")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")
      
    # save model
    if os.path.exists(save_model):
        save_last_model(model, save_model, name)

    print("Final model trained and saved.")
