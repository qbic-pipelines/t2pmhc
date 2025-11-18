import pandas as pd



# ============================================================================= #
#                               FEATURES                                        #
# ============================================================================= #


# Hydrophobicity scale (Kyte-Doolittle values)
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLU': -3.5, 'GLN': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

# electrostatic charges
AA_CHARGES = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLU': -1, 'GLN': 0, 'GLY': 0, 'HIS': 0, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
}

# Atchley factors (from https://bsolomon.us/post/2018/01/13/TCR_analysis_with_Atchyley_factors.html)
ATCHLEY_FACTORS = {
    'ALA': [-0.59145974, -1.30209266, -0.7330651, 1.5703918, -0.14550842],
    'CYS': [-1.34267179, 0.46542300, -0.8620345, -1.0200786, -0.25516894],
    'ASP': [1.05015062, 0.30242411, -3.6559147, -0.2590236, -3.24176791],
    'GLU': [1.35733226, -1.45275578, 1.4766610, 0.1129444, -0.83715681],
    'PHE': [-1.00610084, -0.59046634, 1.8909687, -0.3966186, 0.41194139],
    'GLY': [-0.38387987, 1.65201497, 1.3301017, 1.0449765, 2.06385566],
    'HIS': [0.33616543, -0.41662780, -1.6733690, -1.4738898, -0.07772917],
    'ILE': [-1.23936304, -0.54652238, 2.1314349, 0.3931618, 0.81630366],
    'LYS': [1.83146558, -0.56109831, 0.5332237, -0.2771101, 1.64762794],
    'LEU': [-1.01895162, -0.98693471, -1.5046185, 1.2658296, -0.91181195],
    'MET': [-0.66312569, -1.52353917, 2.2194787, -1.0047207, 1.21181214],
    'ASN': [0.94535614, 0.82846219, 1.2991286, -0.1688162, 0.93339498],
    'PRO': [0.18862522, 2.08084151, -1.6283286, 0.4207004, -1.39177378],
    'GLN': [0.93056541, -0.17926549, -3.0048731, -0.5025910, -1.85303476],
    'ARG': [1.53754853, -0.05472897, 1.5021086, 0.4403185, 2.89744417],
    'SER': [-0.22788299, 1.39869991, -4.7596375, 0.6701745, -2.64747356],
    'THR': [-0.03181782, 0.32571153, 2.2134612, 0.9078985, 1.31337035],
    'VAL': [-1.33661279, -0.27854634, -0.5440132, 1.2419935, -1.26225362],
    'TRP': [-0.59533918, 0.00907760, 0.6719274, -2.1275244, -0.18358096],
    'TYR': [0.25999617, 0.82992312, 3.0973596, -0.8380164, 1.51150958]
}


# tcrblosum for AA types
def read_in_tcrblosum(file_path):
    return pd.read_csv(file_path, sep="\t", index_col=0)

def get_aa_type_tcrblosum(aa, tcrblosum):
    return tcrblosum.loc[aa].tolist()



# feature assigning nodes to complex (one of: tcr, peptide, mhc)
def create_index_list(l, index, chain):
    """
    sets the first and last index for a specific chain.
    """
    if index > 0:
        index += 1    
    l.append(index)
    index += len(chain) - 1
    l.append(index)
    return l, index

def create_complex_list(file_df):
    """
    creates a list of lists. Each list holding the first and last index of a specific complex in the pdb file
    [[hla],[peptide],[tcr_a],[tcr_b]] 
    """
    residue_seq = file_df["target_chainseq"].values[0]
    x = residue_seq.split("/")
    index_counter = 0
    complex_list = [[], [], [], []]
    
    for i, j in enumerate(x):
        complex_list[i], index_counter = create_index_list(complex_list[i], index_counter, j)

    return complex_list

def annotate_residue_with_complex_info(complex_list, index):
    """
    annotate an AA with information to which complex it belongs to
    0: HLA
    1: peptide
    2: TCR_A
    3: TCR_B
    """
    for i, (start, end) in enumerate(complex_list):
        if start <= index <= end:
            return i
    raise ValueError(f"Index {index} not found in any complex")
        
# feature to annotate cdr3 / peptide region specifically
def get_sequence_coord(file_df, sequence):
    """
    function to find cdr3 or peptide sequences
    """
    # find cdr3 coords
    seq = file_df[sequence].values[0]
    # make sure to replace the "/"
    start = file_df["target_chainseq"].values[0].replace("/","").find(seq)
    end = start + len(seq) - 1

    return [start, end]


def annotate_sequence(coords, residues):
    """
    annotates cdr3/peptide information. Done for both a & b & peptide
    0: not cdr3 (a/b) or peptide
    1: cdr3 (a/b) or peptide
    """
    complex_list = []
    for i, res in enumerate(residues):
        if coords[0] <= i <= coords[1]:
            complex_list.append(1)
        else: complex_list.append(0)
    return complex_list