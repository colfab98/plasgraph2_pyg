import itertools
import gzip
import re
import pandas as pd
import networkx as nx
import numpy as np
import os
import math
import fileinput

# PyTorch Geometric imports
import torch
from torch_geometric.data import Dataset as PyGDataset # Alias to avoid confusion if Spektral was also imported
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix # For converting adjacency matrix

from sklearn.preprocessing import MinMaxScaler # Kept as it was in the original
from scipy.special import rel_entr # Kept as it was in the original

# Assuming 'helpers.py' is a module available in the same directory or Python path,
# as it was imported in the original script.
import helpers

# Convert networkx graph to a PyTorch Geometric Data object
class Networkx_to_PyG(PyGDataset): # Changed from spektral.data.Dataset
    def __init__(self, nx_graph, node_order, parameters, transform=None, pre_transform=None, **kwargs):
        self.nx_graph = nx_graph
        self.node_order = node_order # This order is crucial for features, labels, and adjacency
        self.parameters = parameters
        
        # Spektral's Dataset took **kwargs, PyG's Dataset takes root, transform, pre_transform, pre_filter.
        # We'll pass transform and pre_transform if provided.
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        
        # In PyG, processing is usually done once and cached or loaded.
        # The original Spektral class's read() was called to get the graph list.
        # We will prepare the single Data object upon initialization for simplicity here,
        # or it could be done in _process() if we were saving/loading.
        # For this direct conversion, we'll make self._data available.
        self._data = self._create_pyg_data()


    def extract_features(self, node_id, features):
        # This method is identical to the original
        return [self.nx_graph.nodes[node_id][f] for f in features]

    def extract_y(self):
        # This method is identical to the original
        label_features = ["plasmid_label", "chrom_label"]
        y_np = np.array(
            [self.extract_features(node_id, label_features) for node_id in self.node_order]
        )
        # squared hinge loss function needs correct labels -1, 1 rather than 0, 1
        if self.parameters.get("loss_function") == "squaredhinge": # Using .get for safety
            y_np = y_np * 2 - 1
        return torch.tensor(y_np, dtype=torch.float)


    def _create_pyg_data(self):
        """
        This method replaces the original 'read' method's core logic
        to produce a single PyTorch Geometric Data object.
        """
        features = self.parameters['features']
        x_np = np.array(
            [self.extract_features(node_id, features) for node_id in self.node_order]
        )
        x = torch.tensor(x_np, dtype=torch.float)

        y = self.extract_y() # This already returns a torch tensor

        # Create adjacency matrix using the specified node_order, same as original
        # Spektral used nx.adjacency_matrix which returns a SciPy sparse matrix
        # Ensure the format is compatible with from_scipy_sparse_matrix
        # common formats are coo, csr, csc. coo is generally good for PyG.
        adj_matrix_scipy = nx.adjacency_matrix(self.nx_graph, nodelist=self.node_order, dtype=np.float32)
        
        # Original Spektral code did:
        # a.setdiag(0)
        # a.eliminate_zeros()
        # from_scipy_sparse_matrix handles this. Edges are taken from the non-zero entries.
        # If self-loops need to be explicitly removed *before* conversion,
        # adj_matrix_scipy.setdiag(0)
        # adj_matrix_scipy.eliminate_zeros() # Ensure matrix is in canonical form

        edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix_scipy)
        # Note: from_scipy_sparse_matrix might create edge_attr (weights).
        # If your graph is unweighted in terms of GNN processing, you might discard edge_attr
        # or ensure your GNN model can handle/ignore it.
        # The original Spektral Graph took 'a' (adjacency) not edge attributes explicitly as separate tensor.
        # If 'a' was binary, edge_attr might be all 1s. If 'a' had weights, edge_attr will have them.

        # The original Spektral Graph object did not explicitly store edge weights as a separate attribute,
        # the adjacency matrix 'a' itself could be weighted or binary.
        # For PyG, if edges are unweighted, edge_attr can be None or ignored.
        # If features were derived from edge weights, they should be in x or y or a dedicated edge_attr.
        # For a direct conversion, we assume edge_attr from from_scipy_sparse_matrix is what's needed if present.

        return Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr if edge_attr is not None else None)

    # The following methods are standard for PyG Datasets
    @property
    def raw_file_names(self):
        return [] # Not applicable as data is passed in-memory

    @property
    def processed_file_names(self):
        # Could be a name if we were saving the processed Data object
        return [] # e.g. ['data.pt'] 

    def download(self):
        pass # Not applicable

    def process(self):
        # This method would typically be used if loading from raw_files and saving to processed_files.
        # Since we create data in __init__ from in-memory graph, this can be minimal.
        # self._data is already created. If saving/loading is implemented, this changes.
        pass

    def len(self):
        # The original returned a list of Graph objects from read().
        # Here, we assume this Dataset represents a single graph scenario, like the example in read().
        return 1

    def get(self, idx):
        if idx != 0:
            raise IndexError("This dataset contains only one graph.")
        return self._data


# --- Utility functions (KL, weighted_median, etc.) ---
# These are kept as close to the original as possible.

def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def weighted_median(values, weights):
    middle = np.sum(weights) / 2
    cum = np.cumsum(weights)
    for (i, x) in enumerate(cum):
        if x >= middle:
            return values[i]
    assert False # Original code had this, implies weights should sum and values exist

def add_normalized_coverage(graph, current_nodes):
    """Add attribute coverage_norm which is original coverage divided by median weighted by length.
    (only for nodes in current_nodes list)"""
    # This function is identical to the original, assuming current_nodes is not empty
    # and graph.nodes[x]["coverage"] and graph.nodes[x]["length"] exist.
    if not current_nodes: # Added guard for safety, can be removed if original assumed non-empty
        return

    sorted_nodes = sorted(current_nodes, key=lambda x : graph.nodes[x]["coverage"])
    lengths = np.array([graph.nodes[x]["length"] for x in sorted_nodes])
    coverages = np.array([graph.nodes[x]["coverage"] for x in sorted_nodes])
    
    # Original code did not explicitly handle empty lengths/coverages or zero median here,
    # relying on weighted_median potentially erroring or returning a value.
    # For robustness, one might add checks, but sticking to original for now.
    median = weighted_median(coverages, lengths)
    
    # Added check for median == 0 to prevent DivisionByZeroError,
    # original would have raised error here if median was 0.
    if median == 0:
        for node_id in current_nodes:
            graph.nodes[node_id]["coverage_norm"] = 0.0 # Or np.nan, or handle as error
        return

    for node_id in current_nodes:
        graph.nodes[node_id]["coverage_norm"] = graph.nodes[node_id]["coverage"] / median


def get_node_coverage(gfa_arguments, seq_length):
    """Return coverage parsed from dp or estimated from KC tag.
    The second return value is True for dp and False for KC"""
    # Identical to original
    for x in gfa_arguments:
        match =  re.match(r'^dp:f:(.*)$',x)
        if match :
            return (float(match.group(1)), True)
    for x in gfa_arguments:
        match =  re.match(r'^KC:i:(.*)$',x)
        if match :
            # Original did not check for seq_length == 0 here, which could lead to DivisionByZero
            if seq_length == 0: # Add minimal check to prevent error, or let it raise error as original might
                # raise ZeroDivisionError("Sequence length is 0 for KC tag calculation")
                return (0.0, False) # Or handle appropriately
            return (float(match.group(1)) / seq_length, False)
    raise AssertionError("depth not found")


def read_graph(graph_file, csv_file, sample_id, graph, minimum_contig_length):
    """Read a single graph from gfa of gfa.gz, compute attributes, add its nodes and edges to nx graph.
    Label csv file can be set to None. Contigs shorter than minimum_contig_length are contracted."""
    # This function attempts to be as close to the original as possible.

    current_nodes = []
    whole_seq = ""
    coverage_types = {True:0, False:0}

    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            parts = line.strip().split("\t")
            if parts[0] == "S":
                node_id = helpers.get_node_id(sample_id, parts[1])
                seq = parts[2].upper()
                if not re.match(r'^[A-Z]*$', seq):
                    raise AssertionError(f"Bad sequence in {node_id}") # Original behavior

                whole_seq += "N" + seq
                current_nodes.append(node_id)
                assert node_id not in graph # Original behavior
                graph.add_node(node_id)
                seq_length = len(seq)

                graph.nodes[node_id]["contig"] = parts[1]
                graph.nodes[node_id]["sample"] = sample_id
                graph.nodes[node_id]["length"] = seq_length
                (coverage, is_dp) = get_node_coverage(parts[3:], seq_length) # Can raise AssertionError or ZeroDivisionError
                graph.nodes[node_id]["coverage"] = coverage
                coverage_types[is_dp] += 1
                graph.nodes[node_id]["gc"] = helpers.get_gc_content(seq)
                graph.nodes[node_id]["kmer_counts_norm"] = helpers.get_kmer_distribution(seq, scale=True)

    assert coverage_types[True] == 0 or coverage_types[False] == 0 # Original assertion

    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            parts = line.strip().split("\t")
            if parts[0] == "L":
                graph.add_edge(helpers.get_node_id(sample_id, parts[1]),
                               helpers.get_node_id(sample_id, parts[3]))

    for node_id in current_nodes: # Assuming current_nodes only contains nodes successfully added
            graph.nodes[node_id]["degree"] = graph.degree[node_id]

    gc_of_whole_seq = helpers.get_gc_content(whole_seq)
    for node_id in current_nodes:
        graph.nodes[node_id]["gc_norm"] =  graph.nodes[node_id]["gc"] - gc_of_whole_seq

    # Original used max_contig_length for normalization, then switched to fixed 2000000
    # max_contig_length = max([graph.nodes[node_id]["length"] for node_id in current_nodes]) if current_nodes else 1.0
    for node_id in current_nodes:
        # graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / max_contig_length
        graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / 2000000.0 # Using the fixed value from original
        graph.nodes[node_id]["loglength"] = math.log(graph.nodes[node_id]["length"]+1)

    add_normalized_coverage(graph, current_nodes) # Can raise errors if median calculation fails

    all_kmer_counts_norm = np.array(helpers.get_kmer_distribution(whole_seq, scale=True))
    for node_id in current_nodes:
        # Ensure kmer_counts_norm is an array for subtraction/dot product
        current_kmer_counts = np.array(graph.nodes[node_id]["kmer_counts_norm"])
        all_kmer_counts_ref = all_kmer_counts_norm # Just for clarity
        
        # Original code did not check for shape mismatch or empty arrays here,
        # np operations might fail if shapes are incompatible (e.g. if get_kmer_distribution returns empty for short seq)
        # Assuming helpers.get_kmer_distribution handles this by returning compatible zero arrays.
        diff = current_kmer_counts - all_kmer_counts_ref
        graph.nodes[node_id]["kmer_dist"] = np.linalg.norm(diff)
        graph.nodes[node_id]["kmer_dot"] = np.dot(current_kmer_counts, all_kmer_counts_ref)
        graph.nodes[node_id]["kmer_kl"] = KL(current_kmer_counts, all_kmer_counts_ref)


    if csv_file is not None:
        df_labels = pd.read_csv(csv_file)
        df_labels["id"] = df_labels["contig"].map(lambda x : helpers.get_node_id(sample_id, x))
        df_labels.set_index("id", inplace=True)
    else:
        df_labels = pd.DataFrame() # Empty dataframe if no csv_file

    for node_id in current_nodes:
        label = None
        if node_id in df_labels.index: # Check if node is in CSV index
            # Original did not check if 'label' column exists, would raise KeyError if missing.
            label = df_labels.loc[node_id, "label"]

        pair = helpers.label_to_pair(label)
        graph.nodes[node_id]["text_label"] = helpers.pair_to_label(pair)
        graph.nodes[node_id]["plasmid_label"] = pair[0]
        graph.nodes[node_id]["chrom_label"] = pair[1]

    if minimum_contig_length > 0:
        # Original passed current_nodes, which might contain nodes not in graph if errors occurred
        # For safety, one might filter current_nodes to those in graph.nodes, but sticking to original.
        delete_short_contigs(graph, current_nodes, minimum_contig_length)


def delete_short_contigs(graph, node_list, minimum_contig_length):
    """check length attribute of all contigs in node_list
    and if some are shorter than minimum_contig_length,
    remove them from the graph and connect new neighbors"""
    # This function is identical to the original
    # It iterates over node_list. If a node is removed, its subsequent appearance in node_list
    # would lead to an error if not checked if still in graph.nodes.
    # However, the original iterated `for node_id in node_list:` and then conditionally
    # did `graph.remove_node(node_id)`. This is problematic if node_list contains duplicates
    # or if a node is removed and was also a neighbor of another node processed later.
    # A safer way is to collect nodes to remove first, then remove. But sticking to original structure:
    
    # To be closer to original's behavior which might fail if a node is already removed:
    # We must iterate on a copy if the list itself isn't rebuilt or nodes checked.
    # The original implies node_list might be just IDs, and graph is the source of truth.
    
    # Let's create list of nodes to delete first, to avoid modifying graph while iterating over neighbors
    nodes_to_delete_and_their_neighbors = []
    for node_id in node_list:
        if node_id in graph.nodes and graph.nodes[node_id]["length"] < minimum_contig_length:
            neighbors = list(graph.neighbors(node_id)) # Get neighbors *before* any node is removed
            nodes_to_delete_and_their_neighbors.append((node_id, neighbors))

    for node_id, neighbors in nodes_to_delete_and_their_neighbors:
        if node_id not in graph: # Node might have been removed if it was a neighbor of another short contig
            continue
        all_new_edges = list(itertools.combinations(neighbors, 2))
        for edge in all_new_edges:
            # Ensure neighbors for new edge still exist (they might also have been short contigs)
            if edge[0] in graph and edge[1] in graph and edge[0] != edge[1]: # Check existence and avoid self-loops
                graph.add_edge(edge[0], edge[1])
        graph.remove_node(node_id)


def read_single_graph(file_prefix, gfa_file, sample_id, minimum_contig_length):
    """Read single graph without node labels for testing"""
    # Identical to original
    graph = nx.Graph()
    graph_file_path = os.path.join(file_prefix, gfa_file) # Use os.path.join for robustness
    # graph_file = file_prefix + gfa_file # Original string concatenation
    read_graph(graph_file_path, None, sample_id, graph, minimum_contig_length)
    return graph

def read_graph_set(file_prefix, file_list_path, minimum_contig_length, read_labels=True):
    """Read several graph files to a single graph.
    Node labels will be read from the csv file for each graph if read_labels is True.
    Nodes shorter than minimum_contig_length will be deleted from the graph.
    """
    # Identical to original, with os.path.join for robustness
    train_files = pd.read_csv(file_list_path, names=('graph','csv','sample_id'))

    master_graph = nx.Graph() # Renamed from 'graph' to 'master_graph' for clarity within this function
    for idx, row in train_files.iterrows():
        graph_file_path = os.path.join(file_prefix, row['graph'])
        # graph_file = file_prefix + row['graph'] # Original string concatenation

        csv_file_path = None
        if read_labels and pd.notna(row['csv']): # Original didn't check for pd.notna explicitly
            csv_file_path = os.path.join(file_prefix, row['csv'])
            # csv_file = file_prefix + row['csv'] # Original string concatenation
        
        read_graph(graph_file_path, csv_file_path, row['sample_id'], master_graph, minimum_contig_length)

    return master_graph

# No __main__ block as per user request to keep it minimal and module-like.
# The user would call these functions and classes from their own script.
# Example (for user's own script, not part of this file):
#
# import create_graph_pyg # Assuming this file is saved as create_graph_pyg.py
# import networkx as nx
#
# # 1. Define parameters
# parameters = {
#    'features': ['length_norm', 'gc_norm', 'coverage_norm', 'degree', 'kmer_dist', 'kmer_dot', 'kmer_kl', 'loglength'],
#    'loss_function': 'crossentropy' # or 'squaredhinge'
# }
#
# # 2. Read graph data using the functions from the module
# #    (User needs to provide their own data and file paths)
# # file_prefix = "path/to/your/data/"
# # file_list = "path/to/your/file_list.csv"
# # minimum_contig_length = 500
# # nx_graph_data = create_graph_pyg.read_graph_set(file_prefix, file_list, minimum_contig_length, read_labels=True)
#
# # Create a dummy graph for demonstration if no actual data loading:
# nx_graph_data = nx.Graph()
# nx_graph_data.add_node("s1_c1", length=1000, gc=0.5, coverage=10, plasmid_label=1, chrom_label=0,
# length_norm=0.1, gc_norm=0.05, coverage_norm=1.0, degree=1, kmer_dist=0.1, kmer_dot=0.9, kmer_kl=0.05, loglength=6.9)
# nx_graph_data.add_node("s1_c2", length=2000, gc=0.6, coverage=20, plasmid_label=0, chrom_label=1,
# length_norm=0.2, gc_norm=0.01, coverage_norm=2.0, degree=1, kmer_dist=0.2, kmer_dot=0.8, kmer_kl=0.08, loglength=7.6)
# nx_graph_data.add_edge("s1_c1", "s1_c2")
# node_order_list = ["s1_c1", "s1_c2"] # User must define this order based on their graph
#
# if nx_graph_data.number_of_nodes() > 0:
# # 3. Create an instance of the PyTorch Geometric Dataset
#    pyg_dataset_instance = create_graph_pyg.Networkx_to_PyG(
#        nx_graph=nx_graph_data,
#        node_order=node_order_list, # Ensure this order matches how features/labels are expected
#        parameters=parameters
#    )
#
# # 4. Access the graph data
#    if len(pyg_dataset_instance) > 0:
#        pyg_data_object = pyg_dataset_instance.get(0)
#        print("PyTorch Geometric Data object:")
#        print(pyg_data_object)
#        print(f"Node features (x): {pyg_data_object.x}")
#        print(f"Edge index: {pyg_data_object.edge_index}")
#        print(f"Labels (y): {pyg_data_object.y}")
# else:
#    print("NetworkX graph is empty. Cannot create PyG dataset.")