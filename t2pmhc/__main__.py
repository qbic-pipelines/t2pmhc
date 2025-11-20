import os
import numpy as np
import random
import json
import logging
import torch
from pathlib import Path
import rich_click as click

from models.t2pmhc_gcn import train_gcn
from models.t2pmhc_gat import train_gat

from predict.predict_binding import predict_binding

from utils.create_t2pmhc_graphs import create_graphs

from utils.helpers import read_hyperparams


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
#                               HELPERS                                         #
# ============================================================================= #

def load_defaults(json_path: str):
    """Load defaults and resolve all file paths relative to the JSON file."""
    
    json_path = Path(json_path).resolve()          # full path to defaults.json
    base_dir = json_path.parent                    # directory containing defaults.json

    with open(json_path, "r") as f:
        config = json.load(f)

    resolved = {}

    for mode, settings in config.items():
        resolved[mode] = {}
        for key, rel_path in settings.items():
            abs_path = (base_dir / rel_path).resolve()
            resolved[mode][key] = str(abs_path)

    return resolved

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
    type=click.Choice(['t2pmhc-gcn', 't2pmhc-gat']),
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
    type=click.Choice(['t2pmhc-gcn', 't2pmhc-gat']),
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
    type=str,
    required=False,
    help="Path to hyperparams json"
)

@click.option(
    '--model',
    type=str,
    required=False,
    help="t2pmhc model to use for prediction"
)

@click.option(
    '--pae_scaler_structure',
    type=str,
    required=False,
    help="Path to PAE scaler file of the whole structure"
)

@click.option(
    '--pae_scaler_tcrpmhc',
    type=str,
    required=False,
    help="Path to PAE scaler file of the TCR-pMHC complex"
)

@click.option(
    '--hydro_scaler',
    type=str,
    required=False,
    help="Path to hydro scaler"
)

@click.option(
    '--distance_scaler',
    type=str,
    required=False,
    help="Path to distance scaler"
)

@click.option(
    '--pae_scaler_edge',
    type=str,
    required=False,
    help="Path to PAE edge scaler. Only required for GAT"
)

def t2pmhc_predict_binding(mode, samplesheet, saved_graphs, out, hyperparams, model, pae_scaler_structure, pae_scaler_tcrpmhc, hydro_scaler, distance_scaler, pae_scaler_edge):
    """
    Predict TCR-pMHC binding using the t2pmhc models
    """

    # load default config json
    defaults = load_defaults(Path(__file__).parent / "utils" / "t2pmhc_binder_defaults.json")
    # set defaults
    mode_defaults = defaults[mode]

    # Apply defaults only where user did NOT set a value
    hyperparams = hyperparams or mode_defaults.get("hyperparams")
    model = model or mode_defaults.get("model")
    pae_scaler_structure = pae_scaler_structure or mode_defaults.get("pae_scaler_structure")
    pae_scaler_tcrpmhc = pae_scaler_tcrpmhc or mode_defaults.get("pae_scaler_tcrpmhc")
    hydro_scaler = hydro_scaler or mode_defaults.get("hydro_scaler")
    distance_scaler = distance_scaler or mode_defaults.get("distance_scaler")
    if mode == "t2pmhc-gat":
        pae_scaler_edge = pae_scaler_edge or mode_defaults.get("pae_scaler_edge")


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