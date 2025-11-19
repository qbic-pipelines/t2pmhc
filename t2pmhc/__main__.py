import pandas as pd
import numpy as np
import random
import json

import logging

import torch

from models.t2pmhc_gcn import train_gcn
from models.t2pmhc_gat import train_gat

from predict.predict_binding import predict_binding

from utils.create_t2pmhc_graphs import create_graphs

from utils.helpers import read_hyperparams

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

# train gcn
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

# train gat
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


# create graphs
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
    """
    Build t2pmhc graphs from pdb files
    """

    logging.info(f"Building {mode} graphs")
    create_graphs(mode,
              samplesheet,
              out
                )       

    logger.info(".................. done ..................")

# predict binding
@t2pmhc_cli.command()
@click.option(
    '--mode',
    type=str,
    required=True,
    help="Model to use for binding prediction [t2pmhc-gcn, t2pmhc-gat]"
)

@click.option(
    '--samplesheet',
    type=str,
    required=True,
    help="Path to t2pmhc samplesheet"
)

@click.option(
    '--saved_graphs',
    type=str,
    required=True,
    help="Path to saved test graphs"
)

@click.option(
    '--out',
    type=str,
    required=True,
    help="Path to store t2pmhc result tsv"
)

@click.option(
    '--hyperparams',
    default="",
    type=str,
    required=False,
    help="Path to hyperparams json"
)

@click.option(
    '--model',
    default="",
    type=str,
    required=False,
    help="t2pmhc model to use for prediction"
)

@click.option(
    '--pae_scaler_structure',
    default="",
    type=str,
    required=False,
    help="Path to PAE scaler file of the whole structure"
)

@click.option(
    '--pae_scaler_tcrpmhc',
    default="",
    type=str,
    required=False,
    help="Path to PAE scaler file of the TCR-pMHC complex"
)

@click.option(
    '--hydro_scaler',
    default="",
    type=str,
    required=False,
    help="Path to hydro scaler. Only required for GAT"
)

@click.option(
    '--distance_scaler',
    default="",
    type=str,
    required=False,
    help="Path to distance scaler. Only required for GAT"
)

@click.option(
    '--pae_scaler_edge',
    default="",
    type=str,
    required=False,
    help="Path to PAE edge scaler. Only required for GAT"
)

def t2pmhc_predict_binding(mode, samplesheet, saved_graphs, out, hyperparams, model, pae_scaler_structure, pae_scaler_tcrpmhc, hydro_scaler, distance_scaler, pae_scaler_edge):
    """
    Predict TCR-pMHC binding using the t2pmhc models
    """

    print("Predicting binder status")
    predict_binding(mode,
                    samplesheet,
                    saved_graphs,
                    out,
                    hyperparams,
                    model,
                    pae_scaler_structure,
                    pae_scaler_tcrpmhc,
                    hydro_scaler,
                    distance_scaler,
                    pae_scaler_edge)
    
    logger.info(".................. done ..................")
    

    


if __name__ == "__main__":
    train_t2pmhc()