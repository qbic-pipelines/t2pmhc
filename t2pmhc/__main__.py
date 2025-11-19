import pandas as pd
import numpy as np
import random
import json

import logging

import torch

from models.t2pmhc_gcn import train_gcn
from models.t2pmhc_gat import train_gat

from utils.create_t2pmhc_graphs import create_graphs

import rich_click as click

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
#                          create logger                                        #
# ============================================================================= #
# Create logger
logger = logging.getLogger("t2pmhc")
# Create console handler
ch = logging.StreamHandler()
# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False

# ============================================================================= #
#                             MODEL FUNCTIONS                                   #
# ============================================================================= #

def read_hyperparams(json_path):
    if not json_path.endswith(".json"):
        print("Hyperparameters must be in json format")
    
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

def train_t2pmhc():
    print("-------------------------------------------")
    print(" _   ____                  _          ")
    print("\n| |_|___ \ _ __  _ __ ___ | |__   ___")
    print("\n| __| __) | '_ \| '_ ` _ \| '_ \ / __|")
    print("\n| |_ / __/| |_) | | | | | | | | | (__ ")
    print("\n \__|_____| .__/|_| |_| |_|_| |_|\___|")
    print("\n          |_|                         ")
    print("-------------------------------------------")

    t2pmhc_cli()


@click.group()
def t2pmhc_cli():
    """
    t2pmhc: A Structure-Informed Graph Neural Network for Predicting TCR-pMHC Binding 
    """

@t2pmhc_cli.command()
@click.option(
    '--run_name',
    type=str,
    required=True,
    help="Name of the run under which graphs and model will be saved"
)

@click.option(
    '--hyperparameters',
    type=str,
    required=True,
    help="Path to json file containing the hyperparameters"
)

@click.option(
    '--saved_graphs',
    type=str,
    required=True,
    help="Path to the saved graphs"
)

@click.option(
    '--samplesheet',
    type=str,
    required=True,
    help='Path to metadata'
)

@click.option(
    '--save_model',
    type=str,
    required=True,
    help='Directory to save model in'
)

def train_t2pmhc_gcn(samplesheet, run_name, hyperparameters, saved_graphs, save_model):
    """
    t2pmhc-gcn. A Graph Convolutional Network to predict TCR-pMHC binding
    """

    # read in hyperparameters
    hyperparams = read_hyperparams(hyperparameters)

    logging.info("training t2pmhc-gcn")
    train_gcn(samplesheet,
                run_name,
                hyperparams,
                saved_graphs,
                save_model,
                )
    
    logger.info(".................. done ..................")


@t2pmhc_cli.command()
@click.option(
    '--run_name',
    type=str,
    required=True,
    help="Name of the run under which graphs and model will be saved"
)

@click.option(
    '--hyperparameters',
    type=str,
    required=True,
    help="path to json file containing the hyperparameters"
)

@click.option(
    '--saved_graphs',
    type=str,
    required=True,
    help="Path to the saved graphs"
)

@click.option(
    '--samplesheet',
    type=str,
    required=True,
    help='Path to metadata'
)

@click.option(
    '--save_model',
    type=str,
    required=True,
    help='Directory to save model in'
)
def train_t2pmhc_gat(samplesheet, run_name, hyperparameters, saved_graphs, save_model):
    """
    t2pmhc-gat. A Graph Attention Network to predict TCR-pMHC binding
    """

    # read in hyperparameters
    hyperparams = read_hyperparams(hyperparameters)

    logging.info("training t2pmhc-gat")
    train_gat(samplesheet,
                run_name,
                hyperparams,
                saved_graphs,
                save_model,
                )
    
    logger.info(".................. done ..................")


@t2pmhc_cli.command()
@click.option(
    '--mode',
    type=str,
    required=True,
    help="Model for which to create graphs for [t2pmhc-gcn, t2pmhc-gat]"
)

@click.option(
    '--samplesheet',
    type=str,
    required=True,
    help="Path to t2pmhc samplesheet"
)

@click.option(
    '--out',
    type=str,
    required=True,
    help="Path to store the graphs-file"
)

def create_t2pmhc_graphs(mode, samplesheet, out):

    logging.info(f"Building {mode} graphs")
    create_graphs(mode,
              samplesheet,
              out
                )       

    logger.info(".................. done ..................")


if __name__ == "__main__":
    train_t2pmhc()