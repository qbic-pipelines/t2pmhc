#!/usr/bin/env python3

import os
import pandas as pd
import argparse
import numpy as np
import logging
logger = logging.getLogger(__name__)

import torch
from torch_geometric.data import Data

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from joblib import load


from t2pmhc import read_in_samplesheet

from utils.features import (
                            HYDROPHOBICITY,
                            AA_CHARGES,
                            ATCHLEY_FACTORS,
                            get_aa_type_tcrblosum,
                            create_index_list,
                            create_complex_list,
                            annotate_residue_with_complex_info,
                            get_sequence_coord,
                            annotate_sequence
                            )

from utils.helpers import (calculate_contact_map,
                            plot_predictions_labels,
                            str_to_bool,
                            write_run_to_summarytable,
                            save_last_model,
                            save_last_scalers,
                            plot_pred_probs,
                            plot_category_results, 
                            plot_correctly_predicted_samples,
                            get_device
                            )


# ==================================================================================================
#                                           GCN
# ==================================================================================================


def create_gcn_graph(pdb_file, metadata, threshold):
    # read in pae matrix 
    pae_path = pdb_file.replace(".pdb", "_predicted_aligned_error.npy")
    pae_matrix = np.load(pae_path)

    # build contact map and extract residues and edge features
    contact_map, residues, distances, pae_values = calculate_contact_map(pdb_file, pae_matrix, threshold)

    # Convert features to NumPy arrays (not PyTorch)
    aa_type_features = np.array([get_aa_type_tcrblosum(seq1(res)) for res in residues], dtype=np.float32)
    hydrophobicity_features = np.array([HYDROPHOBICITY.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)
    charge_features = np.array([AA_CHARGES.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)

    # complex affiliation feature
    file_df = metadata[metadata["pdb_file_path"].str.contains(os.path.basename(pdb_file))]

    # save some metadata
    identifier = file_df["identifier"].values[0]
    pae_val = file_df["model_2_ptm_pae"].iloc[0]
    pae_pmhc_tcr = file_df["pmhc_tcr_pae"].iloc[0]

    if len(file_df) > 1:
        print(os.path.basename(pdb_file))
        print("ERROR: multiple hits in metadata for filename")
    elif len(file_df) < 1:
        print(os.path.basename(pdb_file))
        print("ERROR: NO hits in metadata for filename")

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
    Used for multiprocessing
    """
    label = int(pdb_file.split("_")[-1].replace(".pdb", ""))
    
    # Get graph data as NumPy arrays
    node_features, edge_index, edge_features, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file = create_gcn_graph(pdb_file, metadata, threshold)

    # Return raw data (not PyTorch objects)
    return node_features, edge_index, edge_features, label, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file


def gcn_create_graphs(pdb_files, metadata, sample_size, threshold, load_graphs, name, saved_graphs, store_graphs, test_run, graphs_path):
    if load_graphs:
        if os.path.exists(saved_graphs):
            print("Loading Graphs from pt file")
            dataset = torch.load(saved_graphs, weights_only=False)
        else:
            print("Error: Saved graphs file does not exist. Please ensure the file path is correct or set 'load_graphs' to False to generate graphs.")
    else:
        dataset = []

        # sample if needed
        if sample_size > 2 and sample_size < np.inf: # inf is default if not chosen (== full dataset)
            pdb_files = np.random.choice(pdb_files, sample_size, replace=False)

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
            print("creating PyTorch Data Instances")
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

            print(f"Processed {i + batch_size} / {len(pdb_files)} files")

        if store_graphs:
            print("saving graphs for reusage")
            if test_run:
                print(f"saving test graphs to {graphs_path}")
                torch.save(dataset, graphs_path)
            else:
                torch.save(dataset, "../utils/graphs_py.pt")
                torch.save(dataset, f"../utils/graph_db/cmap/{name}_graphs.pt")

    return dataset, len(dataset)

# ==================================================================================================
#                                           GAT
# ==================================================================================================

def create_gat_graph(pdb_file, metadata, threshold):
    # read in pae matrix 
    pae_path = pdb_file.replace(".pdb", "_predicted_aligned_error.npy")
    pae_matrix = np.load(pae_path)

    # build contact map and extract residues and edge features
    contact_map, residues, contact_types, distances, pae_values = calculate_contact_map(pdb_file, pae_matrix, threshold)

    # Convert features to NumPy arrays (not PyTorch)
    aa_type_features = np.array([get_aa_type_tcrblosum(seq1(res)) for res in residues], dtype=np.float32)
    hydrophobicity_features = np.array([HYDROPHOBICITY.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)
    charge_features = np.array([AA_CHARGES.get(res, 0.0) for res in residues], dtype=np.float32).reshape(-1, 1)

    # complex affiliation feature
    file_df = metadata[metadata["pdb_file_path"].str.contains(os.path.basename(pdb_file))]

    # save some metadata
    identifier = file_df["identifier"].values[0]
    pae_val = file_df["model_2_ptm_pae"].iloc[0]
    pae_pmhc_tcr = file_df["pmhc_tcr_pae"].iloc[0]

    if len(file_df) > 1:
        print(os.path.basename(pdb_file))
        print("ERROR: multiple hits in metadata for filename")
    elif len(file_df) < 1:
        print(os.path.basename(pdb_file))
        print("ERROR: NO hits in metadata for filename")

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
    Used for multiprocessing
    """
    label = int(pdb_file.split("_")[-1].replace(".pdb", ""))
    
    # Get graph data as NumPy arrays
    node_features, edge_index, edge_features, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file = create_gat_graph(pdb_file, metadata, threshold)

    # Return raw data (not PyTorch objects)
    return node_features, edge_index, edge_features, label, identifier, pae_val, pae_pmhc_tcr, hydrophobicity_features, pdb_file

def gat_create_graphs(pdb_files, metadata, sample_size, threshold, load_graphs, name, saved_graphs, store_graphs, test_run, graphs_path):
    """
    """
    if load_graphs:
        if os.path.exists(saved_graphs):
            print("Loading Graphs from pt file")
            dataset = torch.load(saved_graphs, weights_only=False)
        else:
            print("Error: Saved graphs file does not exist. Please ensure the file path is correct or set 'load_graphs' to False to generate graphs.")
    else:
        dataset = []

        # sample if needed
        if sample_size > 2 and sample_size < np.inf: # inf is default if not chosen (== full dataset)
            pdb_files = np.random.choice(pdb_files, sample_size, replace=False)

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
            print("creating PyTorch Data Instances")
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

            print(f"Processed {i + batch_size} / {len(pdb_files)} files")

        if store_graphs:
            print("saving graphs for reusage")
            if test_run:
                print(f"saving test graphs to {graphs_path}")
                torch.save(dataset, graphs_path)
            else:
                torch.save(dataset, "../utils/graphs_py.pt")
                torch.save(dataset, f"../utils/graph_db/gat/{name}_graphs.pt")

    return dataset, len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Predict binder status of samples in a t2pmhc samplesheets')
    parser.add_argument('--mode', type=str, required=True, help="gcn, gcn-ots, gat")
    parser.add_argument('--samplesheet', type=str, required=True, help="Path to t2pmhc samplesheet")
    parser.add_argument('--out', type=str, required=True, help="Path to store the graphs")

    args = parser.parse_args()

    # init logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    # read in samplesheet 
    pdb_files = read_in_samplesheet(args.samplesheet)
    metadata = pd.read_csv(args.samplesheet, sep="\t")

    # create graphs
    if args.mode == "gat":
        logging.info("Creating Graphs -- gat")
        test_dataset, test_structures = gat_create_graphs(
            pdb_files=pdb_files, metadata=metadata, sample_size=np.inf, threshold=10, load_graphs=False, saved_graphs="", store_graphs=True, name="", test_run=True, graphs_path=args.out
        )
    elif args.mode in ["gcn", "gcn-ots", "gcn-globmean"]:
        logging.info("Creating Graphs -- gcn")
        test_dataset, test_structures = gcn_create_graphs(
            pdb_files=pdb_files, metadata=metadata, sample_size=np.inf, threshold=10, load_graphs=False, saved_graphs="", store_graphs=True, name="",  test_run=True, graphs_path=args.out
        )


if __name__ == "__main__":
    main()
