#!/usr/bin/env python3
"""

This script is used to predict the binding status of a samplesheet using a given t2pmhc model.
It produces a table with binding prediction.

"""

import os
import pandas as pd
import numpy as np
import logging
import torch
from torch_geometric.loader import DataLoader

from joblib import load

from models.t2pmhc_gcn import GCNClassifier, evaluate as gcn_evaluate, create_graph_dataset as gcn_create_graphs
from models.t2pmhc_gat import GATClassifier, evaluate as gat_evaluate, create_graph_dataset as gat_create_graphs

from utils.helpers import read_hyperparams, get_device


logger = logging.getLogger("t2pmhc")


def scale_test(dataset, mode, pae_node_scaler, pae_tcrpmhc_node_scaler, hydro_scaler, distance_scaler, pae_scaler_edge):
    # load shared scalers
    pae_node_scaler = load(pae_node_scaler)
    pae_tcrpmhc_node_scaler = load(pae_tcrpmhc_node_scaler)
    hydro_scaler = load(hydro_scaler)
    # load edge scalers
    distance_scaler = load(distance_scaler)

    if mode == "gcn":
        for graph in dataset:
            pae_val = np.array([[graph.meta["PAE"]]], dtype=np.float32)
            paetcrpmhc_val = np.array([[graph.meta["PAE_TCRpMHC"]]], dtype=np.float32)
            hydro_val = graph.meta["hydro"] # is already an array
            scaled_pae = pae_node_scaler.transform(pae_val)
            scaled_paetcrpmhc = pae_tcrpmhc_node_scaler.transform(paetcrpmhc_val)
            scaled_hydro = hydro_scaler.transform(hydro_val)
            # Add as new feature (column) to node features
            pae_feat = torch.tensor(scaled_pae, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
            paetcrpmhc_feat = torch.tensor(scaled_paetcrpmhc, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
            hydro_feat = torch.tensor(scaled_hydro)
            graph.x = torch.cat([graph.x, pae_feat, paetcrpmhc_feat, hydro_feat], dim=1)
            # edge feature
            edge_features = graph.edge_features
            distances = edge_features[:,0]
            scaled_distances = distance_scaler.transform(distances.reshape(-1, 1))
            graph.edge_attr = torch.tensor(scaled_distances, dtype=torch.float)

            
    elif mode == "gat":
        # load scaler only present in gat
        pae_scaler_edge = load(pae_scaler_edge)
        
        for graph in dataset:
            # get node features
            pae_val = np.array([[graph.meta["PAE"]]], dtype=np.float32)
            paetcrpmhc_val = np.array([[graph.meta["PAE_TCRpMHC"]]], dtype=np.float32)
            hydro_val = graph.meta["hydro"] # is already an array
            # scale features
            scaled_pae = pae_node_scaler.transform(pae_val)
            scaled_paetcrpmhc = pae_tcrpmhc_node_scaler.transform(paetcrpmhc_val)
            scaled_hydro = hydro_scaler.transform(hydro_val)
            # Add as new feature (column) to node features
            pae_feat = torch.tensor(scaled_pae, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
            paetcrpmhc_feat = torch.tensor(scaled_paetcrpmhc, dtype=graph.x.dtype).repeat(graph.x.size(0), 1)
            hydro_feat = torch.tensor(scaled_hydro)
            graph.x = torch.cat([graph.x, pae_feat, paetcrpmhc_feat, hydro_feat], dim=1)
            
            # get edge features
            edge_features = graph.edge_features
            distances = edge_features[:,0]
            paes = edge_features[:,1]
            # scale edge features
            scaled_distances = distance_scaler.transform(distances.reshape(-1, 1))
            scaled_pae = pae_scaler_edge.transform(paes.reshape(-1,1))
            # add edge features as features to the graph
            scaled_features = np.hstack([scaled_distances, scaled_pae]).astype(np.float32)  # Combine distances and PAE values
            graph.edge_attr = torch.tensor(scaled_features, dtype=torch.float)



def add_predictions_to_samplesheet(df, probs, preds, model):

    df["binder_prob"] = probs
    df["binder_prediction"] = preds
    df["model"] = model


def predict_binding(mode, samplesheet, saved_graphs, out, hyperparams, model_path, pae_scaler_structure, pae_scaler_tcrpmhc, hydro_scaler, distance_scaler, pae_scaler_edge):



    # read in hyperparams
    logging.info("Reading Hyperparameters")
    hyperparams = read_hyperparams(hyperparams)
  
    # read in samplesheet
    test_sheet = pd.read_csv(samplesheet, sep="\t")

    # create graphs and init respective model
    if mode == "t2pmhc-gat":
        # read in graphs
        test_dataset, test_structure_count = gat_create_graphs(saved_graphs)
        # scale the test features
        scale_test(test_dataset, "gat", pae_scaler_structure, pae_scaler_tcrpmhc, hydro_scaler, distance_scaler, pae_scaler_edge)
        # init model
        logging.info("Initialising Model")
        model = GATClassifier(input_dim=hyperparams["input_dim"], hidden_dim=hyperparams["hidden_dim"], output_dim=hyperparams["output_dim"], dropout_rate=hyperparams["dropout_rate"], edge_dim=hyperparams["edge_dim"], heads=hyperparams["heads"]
)
    else: 
        logging.info("Creating Graphs")
        # read model and scalers
        test_dataset, test_structure_count = gcn_create_graphs(saved_graphs)
        # scale the test features
        scale_test(test_dataset, "gcn", pae_scaler_structure, pae_scaler_tcrpmhc, hydro_scaler, distance_scaler, "")
        # init model
        model = GCNClassifier(input_dim=hyperparams["input_dim"], hidden_dim=hyperparams["hidden_dim"], output_dim=hyperparams["output_dim"], dropout_rate=hyperparams["dropout_rate"])
    
    # get the device
    device = get_device()
    logging.info(f"Predicting on {device} ...")
    
    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # set at evaluation state

    # set dataloader
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # evaluate the model on dataset
    dummy_criterion  = torch.nn.CrossEntropyLoss() # required by the function

    
    if mode == "t2pmhc-gat": 
        _, _, labels, probs, preds = gat_evaluate(model, loader, dummy_criterion, device, return_probs=True)
    else:
        _, _, labels, probs, preds = gcn_evaluate(model, loader, dummy_criterion, device, return_probs=True)

    logging.info(f"Saving prediction to {out}")
    add_predictions_to_samplesheet(test_sheet, probs, preds, mode)
    
    # save test_sheet
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

    test_sheet.to_csv(out, sep="\t", index=False)



if __name__ == "__main__":
    predict_binding()