"""
THIS SCRIPT CONTAINS HELPER FUNCTIONS USED BY MULTIPLE PYTHON SCRIPTS
"""

import json
import numpy as np
import os
import pandas as pd
import torch
import logging

from Bio.PDB import PDBParser

import joblib

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")


logger = logging.getLogger("t2pmhc")

# ============================================================================= #
#                             MODEL FUNCTIONS                                   #
# ============================================================================= #

def read_hyperparams(json_path):
    """
    Read in hyperparameters from a JSON file.
    Args:
        json_path (str): Path to the JSON file containing hyperparameters.
    Returns:
        dict: Dictionary of hyperparameters.
    """
    if not json_path.endswith(".json"):
        raise FileExistsError("Hyperparameters must be in json format")
    
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    return hyperparams


def read_in_samplesheet(samplesheet):
    """
    Read in a tab-separated sample sheet and extract PDB file paths.
    Args:
        samplesheet (str): Path to the samplesheet file.
    Returns:
        np.ndarray: Array of PDB file paths.
    """
    logger.info("reading samplesheet")
    samplesheet = pd.read_csv(samplesheet, sep="\t")
    try:
        pdb_files = samplesheet["pdb_file_path"].values
    except KeyError:
        raise KeyError("'pdb_file_path' column not found in samplesheet")
        sys.exit(1)
    return pdb_files

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
    Args:
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
    """
    Convert a string to a boolean value.
    Args:
        v (str): Input string.
    Returns:
        bool: Converted boolean value.
    Raises:
        argparse.ArgumentTypeError: If the input string is not 'True' or 'False'.
    """
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else: raise ValueError("String must be 'True' or 'False'")



def get_device():
    """
    Get the available device for PyTorch computations.
    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_last_model(model, model_path, name):
    """
    Saves the state dictionary of a PyTorch model to a specified file.
    Args:
        model (torch.nn.Module): The PyTorch model whose state dictionary is to be saved.
        model_path (str): The path to the directory to store the model in.
        name (str): The name to use for the saved model file (without extension).
    The model is saved to the path '../../data/models/{name}.pt'. Logs a confirmation message upon successful save.
    """
    model_path = f"{model_path}/{name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved final model.")


def save_last_scalers(pae_node_scaler, pae_tcrpmhc_node_scaler, distance_scaler, pae_edge_scaler, hydro_scaler, name, mode, model_path):
    """
    Saves the provided scaler objects to disk using joblib.
    This function serializes and stores two scaler objects, typically used for preprocessing data,
    to the specified directory with filenames based on the provided name parameter.
    Args:
        pae_scaler: The scaler object for the full PAE data to be saved.
        pae_tcrphmc_scaler: The scaler object for the TCR-PMHC PAE data to be saved.
        name (str): The base name to use for the saved scaler files.
    """
    model_path = f"{model_path}/scalers/{name}"
    if not os.path.exists(f"{model_path}/scalers/"):
        os.makedirs(model_path)
    if mode == "GAT":
        joblib.dump(pae_node_scaler, f"{model_path}/{name}_pae_node_FULL.pkl")
        joblib.dump(pae_tcrpmhc_node_scaler, f"{model_path}/{name}_pae_node_TCRPMHC.pkl")
        joblib.dump(distance_scaler, f"{model_path}/{name}_distance.pkl")
        joblib.dump(pae_edge_scaler, f"{model_path}/{name}_pae_edge_FULL.pkl")
        joblib.dump(hydro_scaler, f"{model_path}/{name}_hydro.pkl")
    elif mode == "GCN":
        joblib.dump(pae_node_scaler, f"{model_path}/{name}_pae_node_FULL.pkl")
        joblib.dump(pae_tcrpmhc_node_scaler, f"{model_path}/{name}_pae_node_TCRPMHC.pkl")
        joblib.dump(distance_scaler, f"{model_path}/{name}_distance.pkl")
        joblib.dump(hydro_scaler, f"{model_path}/{name}_hydro.pkl")
    logger.info("Saved PAE scalers")