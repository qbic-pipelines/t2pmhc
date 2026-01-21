#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from Bio.SeqUtils import seq1

import torch
from torch_geometric.data import Data

import multiprocessing as mp

from t2pmhc.utils.features import (
                        HYDROPHOBICITY,
                        AA_CHARGES,
                        ATCHLEY_FACTORS,
                        read_in_tcrblosum,
                        get_aa_type_tcrblosum,
                        create_complex_list,
                        annotate_residue_with_complex_info,
                        get_sequence_coord,
                        annotate_sequence,
                        )

from t2pmhc.utils.helpers import calculate_contact_map, read_in_samplesheet


logger = logging.getLogger("t2pmhc")

# read in tcrblosum
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tcrblosum" / "tcrBLOSUM_all.tsv"
TCRBLOSUM = read_in_tcrblosum(DATA_PATH)


# ==================================================================================================
#                                           GCN
# ==================================================================================================


def create_gcn_graph(pdb_file, metadata, threshold):
    """
    Processes a single PDB file into a graph.
    Args:
        pdb_file (str): Path to the PDB file.
        metadata (pd.DataFrame): Metadata DataFrame.
        threshold (float): Distance threshold for contact map.
    Returns:
        node_features (np.ndarray): Node features array.
        edge_index (np.ndarray): Edge index array.
        edge_features (np.ndarray): Edge features array.
        identifier (str): Identifier from metadata.
        pae_val (float): PAE value from metadata.
        pae_pmhc_tcr (float): PAE TCR-pMHC value from metadata.
        hydrophobicity_features (np.ndarray): Hydrophobicity features array.
        pdb_file (str): Path to the PDB file.
    """
    # read in pae matrix 
    pae_path = pdb_file.replace(".pdb", "_predicted_aligned_error.npy")
    pae_matrix = np.load(pae_path)

    # build contact map and extract residues and edge features
    contact_map, residues, distances, pae_values = calculate_contact_map(pdb_file, pae_matrix, threshold)

    # Convert features to NumPy arrays (not PyTorch)
    aa_type_features = np.array([get_aa_type_tcrblosum(seq1(res), TCRBLOSUM) for res in residues], dtype=np.float32)
    hydrophobicity_features = np.array([HYDROPHOBICITY.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)
    charge_features = np.array([AA_CHARGES.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)

    # complex affiliation feature
    file_df = metadata[metadata["pdb_file_path"].str.contains(os.path.basename(pdb_file))]

    # save some metadata
    identifier = file_df["identifier"].values[0]
    pae_val = file_df["model_2_ptm_pae"].iloc[0]
    pae_pmhc_tcr = file_df["pmhc_tcr_pae"].iloc[0]

    if len(file_df) > 1:
        logger.info(os.path.basename(pdb_file))
        raise AttributeError("ERROR: multiple hits in metadata for filename")
    elif len(file_df) < 1:
        logger.info(os.path.basename(pdb_file))
        raise AttributeError("ERROR: NO hits in metadata for filename")

    complex_list = create_complex_list(file_df)
    complex_features = np.array([annotate_residue_with_complex_info(complex_list, i) for i, res in enumerate(residues)]).reshape(-1, 1)
    cdr3b_coords = get_sequence_coord(file_df, "cdr3b")
    cdr3a_coords = get_sequence_coord(file_df, "cdr3a")
    peptide_coords = get_sequence_coord(file_df, "peptide")
    cdr3b_feature = np.array(annotate_sequence(cdr3b_coords, residues)).reshape(-1, 1)
    cdr3a_feature = np.array(annotate_sequence(cdr3a_coords, residues)).reshape(-1, 1)
    peptide_feature = np.array(annotate_sequence(peptide_coords, residues)).reshape(-1, 1)
    

    # Atchley factors
    atchley_features = np.array([ATCHLEY_FACTORS.get(res, [0.0]*5) for res in residues], dtype=np.float32)

    node_features = np.hstack((aa_type_features, charge_features, atchley_features, complex_features, cdr3b_feature, cdr3a_feature, peptide_feature))

    # Edge features
    edge_index = np.array(np.nonzero(contact_map))  # NumPy array
    # edge_features = np.array(list(zip(distances, pae_values)), dtype=np.float32)  # Combine distances and PAE values
    edge_features = np.array(distances, dtype=np.float32).reshape(-1, 1)

    
    # Return NumPy arrays and meta information
    return node_features, edge_index, edge_features, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file


def process_pdb_file_gcn(pdb_file, metadata, threshold):
    """
    Processes a single PDB file into a graph with a label.
    Used for multiprocessing.
    Args:
        pdb_file (str): Path to the PDB file.
        metadata (pd.DataFrame): Metadata DataFrame.
        threshold (float): Distance threshold for contact map.
    Returns:
        node_features (np.ndarray): Node features array.
        edge_index (np.ndarray): Edge index array.
        edge_features (np.ndarray): Edge features array.
        label (int): Label extracted from the PDB file name.
        identifier (str): Identifier from metadata.
        pae_val (float): PAE value from metadata.
        pae_pmhc_tcr (float): PAE TCR-pMHC value from metadata.
        hydrophobicity_features (np.ndarray): Hydrophobicity features array.
        pdb_file (str): Path to the PDB file.
    """
    label = int(pdb_file.split("_")[-1].replace(".pdb", ""))
    
    # Get graph data as NumPy arrays
    node_features, edge_index, edge_features, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file = create_gcn_graph(pdb_file, metadata, threshold)

    # Return raw data (not PyTorch objects)
    return node_features, edge_index, edge_features, label, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file


def gcn_create_graphs(pdb_files, metadata, threshold, graphs_path):
    """
    Creates GCN graphs from a list of PDB files using multiprocessing.
    Args:
        pdb_files (list): List of PDB file paths.
        metadata (pd.DataFrame): Metadata DataFrame.
        threshold (float): Distance threshold for contact map.
        graphs_path (str): Path to save the resulting graphs.
    Returns:
        dataset (list): List of PyTorch Data objects.
        int: Number of graphs created.
    """
    dataset = []

    # set workers and batches for multiprocessing
    num_workers = min(mp.cpu_count() // 2, 12)
    batch_size = 1000

    for i in range(0, len(pdb_files), batch_size):
        batch_files = pdb_files[i:min(i + batch_size, len(pdb_files))]
        batch_results = []
        
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.starmap(process_pdb_file_gcn, 
                                        [(pdb, metadata, threshold) for pdb in batch_files]):
                batch_results.append(result)

        # Convert NumPy arrays to PyTorch in the main process
        logger.info("creating PyTorch Data Instances")
        for node_features, edge_index, edge_features, label, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file in batch_results:
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            label = torch.tensor([label], dtype=torch.long)

            # save some info as metadata to data object
            meta = {"id" : identifier, "pdb_path" : pdb_file, "PAE" : pae_val, "PAE_TCRpMHC" : pae_pmhc_tcr, "hydro" : hydrophobicity_features}


            # Create a PyG Data object
            data = Data(x=node_features, edge_index=edge_index, y=label)
            # add metadata
            data.meta = meta
            # add edge features
            data.edge_features = edge_features
            dataset.append(data)

        logger.info(f"Processed {i + batch_size} / {len(pdb_files)} files")

    # save the graphs
    logger.info(f"saving test graphs to {graphs_path}")
    dirpath = os.path.dirname(graphs_path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(os.path.dirname(graphs_path))
    torch.save(dataset, graphs_path)

    return dataset, len(dataset)

# ==================================================================================================
#                                           GAT
# ==================================================================================================

def create_gat_graph(pdb_file, metadata, threshold):
    """
    Processes a single PDB file into a graph.
    Args:
        pdb_file (str): Path to the PDB file.
        metadata (pd.DataFrame): Metadata DataFrame.
        threshold (float): Distance threshold for contact map.
    Returns:
        node_features (np.ndarray): Node features array.
        edge_index (np.ndarray): Edge index array.
        edge_features (np.ndarray): Edge features array.
        identifier (str): Identifier from metadata.
        pae_val (float): PAE value from metadata.
        pae_pmhc_tcr (float): PAE TCR-pMHC value from metadata.
        hydrophobicity_features (np.ndarray): Hydrophobicity features array.
        pdb_file (str): Path to the PDB file.
    """
    # read in pae matrix 
    pae_path = pdb_file.replace(".pdb", "_predicted_aligned_error.npy")
    pae_matrix = np.load(pae_path)

    # build contact map and extract residues and edge features
    contact_map, residues, distances, pae_values = calculate_contact_map(pdb_file, pae_matrix, threshold)

    # Convert features to NumPy arrays (not PyTorch)
    aa_type_features = np.array([get_aa_type_tcrblosum(seq1(res), TCRBLOSUM) for res in residues], dtype=np.float32)
    hydrophobicity_features = np.array([HYDROPHOBICITY.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)
    charge_features = np.array([AA_CHARGES.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)

    # complex affiliation feature
    file_df = metadata[metadata["pdb_file_path"].str.contains(os.path.basename(pdb_file))]

    # save some metadata
    identifier = file_df["identifier"].values[0]
    pae_val = file_df["model_2_ptm_pae"].iloc[0]
    pae_pmhc_tcr = file_df["pmhc_tcr_pae"].iloc[0]

    if len(file_df) > 1:
        logger.info(os.path.basename(pdb_file))
        raise AttributeError("ERROR: multiple hits in metadata for filename")
    elif len(file_df) < 1:
        logger.info(os.path.basename(pdb_file))
        raise AttributeError("ERROR: NO hits in metadata for filename")

    complex_list = create_complex_list(file_df)
    complex_features = np.array([annotate_residue_with_complex_info(complex_list, i) for i, res in enumerate(residues)]).reshape(-1, 1)
    cdr3b_coords = get_sequence_coord(file_df, "cdr3b")
    cdr3a_coords = get_sequence_coord(file_df, "cdr3a")
    peptide_coords = get_sequence_coord(file_df, "peptide")
    cdr3b_feature = np.array(annotate_sequence(cdr3b_coords, residues)).reshape(-1, 1)
    cdr3a_feature = np.array(annotate_sequence(cdr3a_coords, residues)).reshape(-1, 1)
    peptide_feature = np.array(annotate_sequence(peptide_coords, residues)).reshape(-1, 1)
    
    
    # Atchley factors
    atchley_features = np.array([ATCHLEY_FACTORS.get(res, [0.0]*5) for res in residues], dtype=np.float32)

    # Concatenate all node features (still NumPy)
    node_features = np.hstack((aa_type_features, charge_features, atchley_features, complex_features, cdr3b_feature, cdr3a_feature, peptide_feature))

    # Edge features
    edge_index = np.array(np.nonzero(contact_map))  # NumPy array
    edge_features = np.array(list(zip(distances, pae_values)), dtype=np.float32)  # Combine distances and PAE values
    
    # Return NumPy arrays and meta information
    return node_features, edge_index, edge_features, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file


def process_pdb_file_gat(pdb_file, metadata, threshold):
    """
    Processes a single PDB file into a graph with a label.
    Used for multiprocessing.
    Args:
        pdb_file (str): Path to the PDB file.
        metadata (pd.DataFrame): Metadata DataFrame.
        threshold (float): Distance threshold for contact map.
    Returns:
        node_features (np.ndarray): Node features array.
        edge_index (np.ndarray): Edge index array.
        edge_features (np.ndarray): Edge features array.
        label (int): Label extracted from the PDB file name.
        identifier (str): Identifier from metadata.
        pae_val (float): PAE value from metadata.
        pae_pmhc_tcr (float): PAE TCR-pMHC value from metadata.
        hydrophobicity_features (np.ndarray): Hydrophobicity features array.
        pdb_file (str): Path to the PDB file.
    """
    label = int(pdb_file.split("_")[-1].replace(".pdb", ""))
    
    # Get graph data as NumPy arrays
    node_features, edge_index, edge_features, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file = create_gat_graph(pdb_file, metadata, threshold)

    # Return raw data (not PyTorch objects)
    return node_features, edge_index, edge_features, label, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file

def gat_create_graphs(pdb_files, metadata, threshold, graphs_path):
    """
    Creates GAT graphs from a list of PDB files using multiprocessing.
    Args:
        pdb_files (list): List of PDB file paths.
        metadata (pd.DataFrame): Metadata DataFrame.
        threshold (float): Distance threshold for contact map.
        graphs_path (str): Path to save the resulting graphs.
    Returns:
        dataset (list): List of PyTorch Data objects.
        int: Number of graphs created.
    """
    dataset = []

    # set workers and batch for multiprocessing
    num_workers = min(mp.cpu_count() // 2, 12)
    batch_size = 1000

    for i in range(0, len(pdb_files), batch_size):
        batch_files = pdb_files[i:min(i + batch_size, len(pdb_files))]
        batch_results = []
        
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.starmap(process_pdb_file_gat, 
                                        [(pdb, metadata, threshold) for pdb in batch_files]):
                batch_results.append(result)

        # Convert NumPy arrays to PyTorch in the main process
        logger.info("creating PyTorch Data Instances")
        for node_features, edge_index, edge_features, label, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file in batch_results:
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            label = torch.tensor([label], dtype=torch.long)

            # save some info as metadata to data object
            meta = {"id" : identifier, "pdb_path" : pdb_file, "PAE" : pae_val, "PAE_TCRpMHC" : pae_pmhc_tcr, "hydro" : hydrophobicity_features}


            # Create a PyG Data object
            data = Data(x=node_features, edge_index=edge_index, y=label)
            # add metadata
            data.meta = meta
            # add edge features 
            data.edge_features = edge_features
            dataset.append(data)

        logger.info(f"Processed {i + batch_size} / {len(pdb_files)} files")

    
    logger.info(f"saving graphs to {graphs_path}")
    torch.save(dataset, graphs_path)

    return dataset, len(dataset)


def create_graphs(mode, samplesheet, out):
    """
    Main function to create graphs based on the specified mode.
    Args:
        mode (str): Mode of graph creation ("t2pmhc-gat" or "t2pmhc-gcn").
        samplesheet (str): Path to the samplesheet file.
        out (str): Output path to save the graphs.
    """
    # read in samplesheet 
    pdb_files = read_in_samplesheet(samplesheet)
    metadata = pd.read_csv(samplesheet, sep="\t")

    # create graphs
    if mode == "t2pmhc-gat":
        #logging.info("Creating Graphs -- t2pmhc-gat")
        logger.info("Creating Graphs -- t2pmhc-gat")
        gat_create_graphs(pdb_files=pdb_files, metadata=metadata, threshold=10, graphs_path=out)
    elif mode == "t2pmhc-gcn":
        #logging.info("Creating Graphs -- t2pmhc-gcn")
        logger.info("Creating Graphs -- t2pmhc-gcn")
        gcn_create_graphs(pdb_files=pdb_files, metadata=metadata, threshold=10, graphs_path=out)


if __name__ == "__main__":
    create_graphs()
