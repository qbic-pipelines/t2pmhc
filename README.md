# t2pmhc
t2pmhc: A Structure-Informed Graph Neural Network for Predicting TCR–pMHC Binding

<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/t2pmhc_logo.png">
    <img alt="tpmhc" src="assets/t2pmhc_logo.png" width="500">
  </picture>
</h1>

# Installation

## 1. Docker

Pull image from DockerHub:

``` docker pull mvp9/t2pmhc:1.0.0 ```

## 2. Python

- Clone the repository

``` git clone https://github.com/qbic-pipelines/t2pmhc/ ```

- cd into the repository

- Create a fresh conda env

``` conda create -n t2pmhc python=3.11 ```

- Install the requirements.txt

``` pip install -r requirements.txt ``` 

- Install *t2pmhc* locally

``` pip install -e . ```

Now you can use the *t2pmhc* anywhere on your machine.

# Usage

## Create pdb files

t2pmhc currently supports pdb files created with [TCRdock](https://github.com/phbradley/TCRdock).  
You can either follow the documentation of tcrdock or use our branch of the [nf-core/proteinfold](https://github.com/nf-core/proteinfold/) pipeline

### TCRDock in nf-core proteinfold


Clone the repository and checkout to the tcrdock branch
1.  ``` git clone https://github.com/mapo9/nf-core_proteinfold ```
2. ``` git checkout -b tcrdock ```

See the [documentation](https://github.com/mapo9/nf-core_proteinfold/tree/tcrdock) to create the docker container and run the pipeline.

Minimal samplesheet:  

```console
organism,mhc_class,mhc,peptide,va,ja,cdr3a,vb,jb,cdr3b,identifier
human,1,A*02:01:48,RLQSLQTYV,TRAV16*01,TRAJ39*01,CALSGFNNAGNMLTF,TRBV11-2*01,TRBJ2-3*01,CASSLGGAGGADTQYF,a2341ad
human,1,A*02:01:48,YLQPRTFLL,TRAV12-2*01,TRAJ30*01,CAVNRDDKIIF,TRBV7-9*01,TRBJ2-7*01,CASSPDIEQYF,a2341ad
```

| Column | Description |
| ---------- | --------------------------------------------------------------------------------------------------- |
| `organism` | 'human'. |
| `mhc_class` | 1 |
| `mhc` | The MHC allele, e.g. 'A\*02:01' |
| `peptide` | The peptide sequence. |
| `va` | V-alpha gene. |
| `ja` | J-alpha gene. |
| `cdr3a` | CDR3-alpha sequence, starts with C, ends with the F/W/etc right before the GXG sequence in the J gene. |
| `vb` | V-beta gene. |
| `jb` | J-beta gene. |
| `cdr3b` | CDR3-beta sequence, starts with C, ends with the F/W/etc right before the GXG sequence in the J gene. |
| `identifier` | Unique sample identifier. |

## Create t2pmhc graphs

To create the graphs expected by the models from the pdb files, you can run the following command:
t2pmhc expects TCRdock output as input for the graph generation step.
Minimal samplesheet:  

```console
organism,mhc_class,mhc,peptide,va,ja,cdr3a,vb,jb,cdr3b,identifier,model_2_ptm_pae,pmhc_tcr_pae,target_chainseq
human,1,A*02:01:48,RLQSLQTYV,TRAV16*01,TRAJ39*01,CALSGFNNAGNMLTF,TRBV11-2*01,TRBJ2-3*01,CASSLGGAGGADTQYF,1sr34,2.43,6.24,CALSGFNNAGNMLTF/RLQSLQTYV/CASSLGGAGGADTQYF
human,1,A*02:01:48,YLQPRTFLL,TRAV12-2*01,TRAJ30*01,CAVNRDDKIIF,TRBV7-9*01,TRBJ2-7*01,CASSPDIEQYF,223dse2,4.5,7.2,YLQPRTFLL/CAVNRDDKIIF/CASSPDIEQYF
```

| Column | Description |
| ---------- | --------------------------------------------------------------------------------------------------- |
| `organism` | 'human'. |
| `mhc_class` | 1 |
| `mhc` | The MHC allele, e.g. 'A\*02:01' |
| `peptide` | The peptide sequence. |
| `va` | V-alpha gene. |
| `ja` | J-alpha gene. |
| `cdr3a` | CDR3-alpha sequence, starts with C, ends with the F/W/etc right before the GXG sequence in the J gene. |
| `vb` | V-beta gene. |
| `jb` | J-beta gene. |
| `cdr3b` | CDR3-beta sequence, starts with C, ends with the F/W/etc right before the GXG sequence in the J gene. |
| `identifier` | Unique sample identifier. |
| `model_2_ptm_pae` | PAE of the complex (provided by TCRdock). |
| `pmhc_tcr_pae` | TCR-pMHC specific PAE value (provided by TCRdock). |
| `target_chainseq` | Full sequence of the complex (MHC/peptide/TCRA/TCRB) (provided by TCRdock). |



```
t2pmhc create-t2pmhc-graphs \
    --mode <tpmhc-gcn,t2pmhc-gat> \
    --samplesheet samplesheet.tsv \
    --out <path/to/graphs.pt> \
```

## Train t2pmhc models

### t2pmhc-gcn

```
t2pmhc train-t2pmhc-gcn \
    --run_name <name to save model under> \
    --hyperparameters t2pmhc/data/hyperparams/t2pmhc_gcn.json \
    --samplesheet samplesheet.tsv \
    --saved_graphs <path/to/graphs.pt> \
    --save_model <path/to/model_dir>
```

### t2pmhc-gat

```
t2pmhc train-t2pmhc-gat \
    --run_name <name to save model under> \
    --hyperparameters t2pmhc/data/hyperparams/t2pmhc_gat.json \
    --samplesheet samplesheet.tsv \
    --saved_graphs <path/to/graphs.pt> \
    --save_model <path/to/model_dir>
```

## Predict binder status of TCR-pMHC samples

You can either use a model you trained or use the published default models to predict the binder status for your TCR-pMHC complexes.  
The resulting tsv file will contain the column **binder_prob** containing the binding probability of the complex assigned by t2pmhc.

### Default mode
```
t2pmhc t2pmhc-predict-binding \
    --mode <t2pmhc-gcn, t2pmhc-gat> \
    --samplesheet samplesheet.tsv \
    --saved_graphs <path/to/graphs.pt> \
    --out samplesheet_predicted.tsv
```

### Retrained mode
```
t2pmhc t2pmhc-predict-binding \
    --mode <t2pmhc-gcn, t2pmhc-gat> \
    --samplesheet samplesheet.tsv \
    --saved_graphs <path/to/graphs.pt> \
    --out samplesheet_predicted.tsv \
    --model <model.pt> \
    --pae_scaler_structure <pae_node_FULL.pkl> \
    --pae_scaler_tcrpmhc <pae_node_TCRPMHC.pkl> \
    --hydro_scaler <hydro_scaler.pkl> \
    --distance_scaler <distance_scaler.pkl> \
    --pae_scaler_edge <pae_edge_FULL.pkl> \
```