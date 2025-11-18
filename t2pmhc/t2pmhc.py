import argparse
import pandas as pd
import numpy as np
import random
import json

import torch

from models.gcn_contactmaps import gcn_contactmap
from models.gat_contactmaps import gat_contactmap
from models.t2pmhc_gcn import train_gcn
from models.t2pmhc_gat import train_gat
from helper_functions.tcr_phla_helpers import plot_loss_auc, plot_predictions_labels, save_metrics, str_to_bool
import sys



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
#                             MODEL FUNCTIONS                                   #
# ============================================================================= #

def read_hyperparams(json_path, parser):
    if not json_path.endswith(".json"):
        parser.error("Hyperparameters must be in json format")
    
    with open(json_path, "r") as f:
        hyperparams = json.load(f)

    return hyperparams


def read_in_samplesheet(samplesheet):
    """
    Read in a tab-separated sample sheet and extract PDB file paths.
    
    Parameters
    ----------
    samplesheet : str
        Path to the tab-separated sample sheet file containing a 'pdb_file_path' column.
        
    Returns
    -------
    numpy.ndarray
        Array of PDB file paths extracted from the sample sheet.
    """
    samplesheet = pd.read_csv(samplesheet, sep="\t")
    try:
        pdb_files = samplesheet["pdb_file_path"].values
    except KeyError:
        print("Error: 'pdb_file_path' column not found in samplesheet")
        sys.exit(1)
    return pdb_files

        


# ============================================================================= #
#                               MAIN FUNCTION                                   #
# ============================================================================= #

def main():
    parser = argparse.ArgumentParser(description='Train a GCN model for TCR-pHLA binding prediction')
    parser.add_argument('--mode', type=str, required=True, help="Mode in which to train the model [t2pmhc_gcn, t2pmhc_gat]")
    parser.add_argument('--run_name', type=str, required=True, help="Name of the run under which graphs and model will be saved")
    parser.add_argument('--hyperparameters', type=str, required=True, help="path to json file containing the hyperparameters")
    parser.add_argument('--samplesheet', type=str, required=True, help='Path to metadata')
    parser.add_argument('--saved_graphs', type=str, required=False, help="Path to saved graphs")
    parser.add_argument('--save_model', type=str, required=True, help='Directory to save model in')
    
    args = parser.parse_args()

    # possible modes
    modes = ["t2pmhc_gcn", "t2pmhc_gat"]

    # read in hyperparameters
    hyperparams = read_hyperparams(args.hyperparameters, parser)

    # get pdb files
    pdb_files = read_in_samplesheet(args.samplesheet)

    # check for angström
    if args.mode in modes:
        if args.angstrom_thrsd is None:
            parser.error(f"angstrom_thrsd is required to calculate Contact Map")

    # check for storing graphs
    if args.store_graphs:
        store_graphs = str_to_bool(args.store_graphs)
    else:
        store_graphs = True

    # check for saved graphs
    if args.saved_graphs is None:
        saved_graphs = ""
    else:
        saved_graphs = args.saved_graphs

    if args.mode == "t2pmhc_gcn":
        train_gcn(args.samplesheet,
                  pdb_files,
                  args.name,
                  hyperparams,
                  saved_graphs,
                  args.save_model
                  )

    elif args.mode == "t2pmhc_gat":
        train_gat(args.samplesheet,
                  pdb_files,
                  args.name,
                  hyperparams,
                  saved_graphs,
                  args.save_model
                  )
        
    else:
        parser.error(f"Mode must be in {modes}")
        

    print(".................. done ..................")


if __name__ == "__main__":
    main()