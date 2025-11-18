import argparse
import pandas as pd
import numpy as np
import random
import json

import logging
logger = logging.getLogger(__name__)

import torch

from models.t2pmhc_gcn import train_gcn
from models.t2pmhc_gat import train_gat

from utils.helpers import str_to_bool

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
    logging.info("reading samplesheet")
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
    parser.add_argument('--saved_graphs', type=str, required=True, help="Path to saved graphs")
    parser.add_argument('--save_model', type=str, required=True, help='Directory to save model in')
    
    args = parser.parse_args()

    # init logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    # possible modes
    modes = ["t2pmhc_gcn", "t2pmhc_gat"]

    # read in hyperparameters
    hyperparams = read_hyperparams(args.hyperparameters, parser)

    # get pdb files
    pdb_files = read_in_samplesheet(args.samplesheet)


    if args.mode == "t2pmhc_gcn":
        logging.info("training t2pmhc-gcn")
        train_gcn(args.samplesheet,
                  args.run_name,
                  hyperparams,
                  args.saved_graphs,
                  args.save_model,
                  )

    elif args.mode == "t2pmhc_gat":
        logging.info("training t2pmhc-gat")
        train_gat(args.samplesheet,
                  args.run_name,
                  hyperparams,
                  args.saved_graphs,
                  args.save_model,
                  )
        
    else:
        parser.error(f"Mode must be in {modes}")
        

    print(".................. done ..................")


if __name__ == "__main__":
    main()