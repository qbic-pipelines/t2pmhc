"""
THIS SCRIPT CONTAINS HELPER FUNCTIONS USED BY MULTIPLE PYTHON SCRIPTS
"""

import argparse
import json
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import torch
import seaborn as sns

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import random
from collections import Counter

import joblib

import wandb
import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")


# ============================================================================= #
#                          Structure Representation                             #
# ============================================================================= #

def calculate_contact_map(pdb_file, pae_matrix, threshold):
    """
    Calculate the contact map of a protein structure from a PDB file.
    This function reads a PDB file, extracts the C-alpha atom coordinates of each residue,
    and calculates the pairwise distances between these atoms. If the distance between 
    two residues is less than or equal to the specified threshold, a contact is recorded 
    in the contact map.
    Parameters:
    pdb_file (str): Path to the PDB file containing the protein structure.
    threshold (float): Distance threshold to determine if two residues are in contact.
    Returns:
    tuple: A tuple containing:
        - contact_map (np.ndarray): A binary matrix indicating contacts between residues.
        - residues (list): A list of residue names corresponding to the rows/columns of the contact map.
        - distances (list): A list of distances between residues that are in contact.
        - pae_values (list): A list of PAE values between residues that are in contact.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('Protein', pdb_file)
    coords = []
    residues = []
    contact_types = []
    distances = []
    paes = []

    # Extract C-alpha atom coordinates and residue names
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].coord)
                    residues.append(residue.resname)

    coords = np.array(coords)
    num_residues = len(coords)
    contact_map = np.zeros((num_residues, num_residues))

    # Calculate pairwise distances to build the contact map
    for i in range(num_residues):
        for j in range(num_residues):
            if i != j:  # Ensure no self-loops
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance <= threshold:
                    contact_map[i, j] = 1  # Binary contact map
                    #contact_types.append(compute_contact_type(residues[i], residues[j], distance))
                    distances.append(distance)
                    paes.append(pae_matrix[i, j])
                    

    return contact_map, residues, distances, paes

# ============================================================================= #
#                                  GENERAL                                      #
# ============================================================================= #

def str_to_bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else: raise argparse.ArgumentTypeError("String must be 'True' or 'False'")



def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ============================================================================= #
#                                  PLOTS                                        #
# ============================================================================= #

def plot_loss_auc(results, labels, probs, fold=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot training and validation loss
    axes[0].plot(results['epochs'], results['train_loss'], label='Train Loss', color='green')
    axes[0].plot(results['epochs'], results['val_loss'], label='Validation Loss', color='orange')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC-AUC')
    axes[1].legend(loc="upper left")

    plt.tight_layout()

    # Save before show
    filename = f"loss_and_auc_fold{fold+1 if fold is not None else ''}.png"
    plt.savefig(filename)
    wandb.log({f"loss_auc_fold{fold+1 if fold is not None else ''}": wandb.Image(filename)})
    plt.close()



def plot_predictions_labels(preds, labels, title):
    # check that it doenst just say everything is negative or positive
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot for test_preds
    unique, counts = np.unique(preds, return_counts=True)
    axes[0].bar(unique, counts, edgecolor='black', width=0.1)
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} Predictions')
    axes[0].set_xticks([0, 1])
    # Plot for test_labels
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    axes[1].bar(unique_labels, counts_labels, edgecolor='black', width=0.1)
    axes[1].set_xlabel('True Class')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Bar Plot of {title} Labels')
    axes[1].set_xticks([0, 1])

    plt.tight_layout()

    plt.savefig("labels.png")

    wandb.log({"labels" : wandb.Image("labels.png")})
    plt.close()



def plot_conf_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-binder', 'Binder'],
                yticklabels=['Non-binder', 'Binder'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    # log to wandb
    filename = "conf_matr.png"
    plt.savefig(filename)
    wandb.log({"conf_matr" : wandb.Image(filename)})
    plt.close()



def plot_pred_probs(labels, probs):
    plt.figure(figsize=(12, 6))
    plt.hist(np.array(probs)[np.array(labels) == 0], bins=50, alpha=0.5, label='Non-binders', density=True)
    plt.hist(np.array(probs)[np.array(labels) == 1], bins=50, alpha=0.5, label='Binders', density=True)
    plt.xlim([0.0, 1.05])
    plt.xlabel('Predicted probability of being a binder')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities by True Class')
    plt.legend()
    
    # log to wandb
    filename = "pred_probs.png"
    plt.savefig(filename)
    wandb.log({"pred_probs" : wandb.Image(filename)})
    plt.close()




def plot_category_results(tn, fp, fn, tp):
    # Create bar plot of prediction results
    plt.figure(figsize=(12, 6))
    metrics = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    values = [tn, fp, fn, tp]
    colors = ['green', 'red', 'red', 'green']

    bars = plt.bar(metrics, values, color=colors)
    plt.ylabel('Count')
    plt.title('Prediction Results by Category')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:,}', ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # log to wandb
    filename = "category_results.png"
    plt.savefig(filename)
    wandb.log({"cat_results" : wandb.Image(filename)})
    plt.close()



def plot_correctly_predicted_samples(tn, fp, fn, tp):
    # Calculate percentages for correct predictions by class
    pos_correct_percent = tp / (tp + fn) * 100  # Same as recall or sensitivity
    neg_correct_percent = tn / (tn + fp) * 100  # Same as specificity

    # Plot percentage of correct predictions by class
    plt.figure(figsize=(10, 6))
    plt.bar(['Negative Class', 'Positive Class'], 
        [neg_correct_percent, pos_correct_percent],
        color=['skyblue', 'lightgreen'])
    plt.ylabel('Percentage Correctly Predicted')
    plt.title('Percentage of Correctly Predicted Samples by Class')
    plt.ylim(0, 100)

    # Add value labels on top of each bar
    for i, v in enumerate([neg_correct_percent, pos_correct_percent]):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # log to wandb
    filename = "correctly_predicted.png"
    plt.savefig(filename)
    wandb.log({"correctly_pred" : wandb.Image(filename)})
    plt.close()



def save_metrics(run_id, mode, preds, labels, hyperparams, structure_count):
    """
    saves predictions and true values of the run to a json
    """
    head_dir = "../../data/model_results"

    # make serializable for json
    preds = [float(p) for p in preds]
    labels = [float(l) for l in labels]

    # Load previous data if exists
    try:
        with open(f"{head_dir}/{mode}_roc_vals.json", "r") as f:
            # Check if file is not empty
            if os.stat(f"{head_dir}/{mode}_roc_vals.json").st_size > 0:
                results = json.load(f)
            else: 
                results = []
    except FileNotFoundError:
        results = []

    # Append new results
    timestamp = datetime.datetime.now().isoformat()
    results.append({"run": run_id, 
                    "timestamp": timestamp, 
                    "true_labels": labels,
                    "predictions": preds,
                    "input_dim" : hyperparams["input_dim"], 
                    "hidden_dim" : hyperparams["hidden_dim"],
                    "output_dim" : hyperparams["output_dim"],  
                    "learning_rate" : hyperparams["learning_rate"],
                    "num_epochs" : hyperparams["num_epochs"],
                    "weight_decay" : hyperparams["weight_decay"],
                    "dropout_rate" : hyperparams["dropout_rate"],
                    "batch_size" : hyperparams["batch_size"],
                    "k" : hyperparams["k"],
                    "sample_size" : structure_count}
                )

    # Convert JSON to string with indentation
    json_str = json.dumps(results, indent=4)

    # Use regex to collapse lists onto one line
    json_str = re.sub(r'\[\s+([\d.,\s]+?)\s+\]', lambda m: "[" + " ".join(m.group(1).split()) + "]", json_str)

    # Save back to JSON
    with open(f"{head_dir}/{mode}_roc_vals.json", "w") as f:
        f.write(json_str)




def write_run_to_summarytable(summary_dict):
    # hardcoded for now
    summary_df = pd.read_csv("../../data/model_results/runs_summary.tsv", sep="\t")
    
    # Ensure all keys in summary_dict are present as columns in the DataFrame
    for key in summary_dict.keys():
        if key not in summary_df.columns:
            summary_df[key] = None

    # Create a new row with the updated DataFrame columns
    new_row = pd.DataFrame([summary_dict])[summary_df.columns]

    # Append the new row to the existing DataFrame
    summary_df = pd.concat([summary_df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the TSV file
    summary_df.to_csv("../../data/model_results/runs_summary.tsv", sep='\t', index=False)

    print("Wrote results to summary table.")



def save_last_model(model, model_path, name):
    """
    Saves the state dictionary of a PyTorch model to a specified file.
    Args:
        model (torch.nn.Module): The PyTorch model whose state dictionary is to be saved.
        model_path (str): The path to the directory to store the model in.
        name (str): The name to use for the saved model file (without extension).
    The model is saved to the path '../../data/models/{name}.pt'. Prints a confirmation message upon successful save.
    """
    model_path = f"{model_path}/{name}.pt"
    torch.save(model.state_dict(), model_path)
    print("Saved final model.")


def save_last_scalers(pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler, name, mode):
    """
    Saves the provided scaler objects to disk using joblib.
    This function serializes and stores two scaler objects, typically used for preprocessing data,
    to the specified directory with filenames based on the provided name parameter.
    Args:
        pae_scaler: The scaler object for the full PAE data to be saved.
        pae_tcrphmc_scaler: The scaler object for the TCR-PMHC PAE data to be saved.
        name (str): The base name to use for the saved scaler files.
    """
    if mode == "GAT":
        joblib.dump(pae_node_scaler, f"../../data/scalers/{name}_pae_node_FULL.pkl")
        joblib.dump(pae_tcrpmhc_node_scaler, f"../../data/scalers/{name}_pae_node_TCRPMHC.pkl")
        joblib.dump(distance_scaler, f"../../data/scalers/{name}_distance.pkl")
        joblib.dump(pae_edge_scaler, f"../../data/scalers/{name}_pae_edge_FULL.pkl")
        joblib.dump(hydro_scaler, f"../../data/scalers/{name}_hydro.pkl")
    elif mode == "GCN":
        joblib.dump(pae_node_scaler, f"../../data/scalers/{name}_pae_node_FULL.pkl")
        joblib.dump(pae_tcrpmhc_node_scaler, f"../../data/scalers/{name}_pae_node_TCRPMHC.pkl")
        joblib.dump(distance_scaler, f"../../data/scalers/{name}_distance.pkl")
        joblib.dump(hydro_scaler, f"../../data/scalers/{name}_hydro.pkl")
    print("Saved PAE scalers")