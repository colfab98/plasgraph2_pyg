#!/usr/bin/env python3
# coding: utf-8

"""
Run plASgraph model on a single gfa file or a whole set of files.

The first argument is a command name, followed by arguments of a particular 
command. Commands are 

  set   runs plASgraph on a set of files
  gfa   runs plASgraph on a single gfa file

More details run test.py [command] -h 
"""

import gzip
import os
import pandas as pd
import networkx as nx
import numpy as np
import argparse
import inspect
import architecture

# from tensorflow import keras

# from spektral.transforms import GCNFilter
# from spektral.data.loaders import SingleLoader

import create_graph as create_graph
import helpers
import config
import torch
from architecture import PlasGraphModel

from torch_geometric.utils import add_self_loops, degree


def main_set(
    test_file_list: 'csv file with a list of testing samples',
    file_prefix: 'common prefix to be used for all filenames listed in test_file_list, e.g. ../data/',
    model_dir: 'name of an input folder with the trained model',
    output_file: 'name of the output csv file'
):
    """Runs a trained model on datasets listed in a file and writes outputs as a csv file.

    The test_file_list is a csv with a list of testing data samples. It contains no header and three comma-separated values:
      name of the gfa.gz file from short read assemby,
      name of the csv file with correct answers (not used by this script, can be arbitrary value)
      id of the sample (string without colons, commas, whitespace, unique within the set)
    """

    test_files = pd.read_csv(test_file_list, names=('graph', 'csv', 'sample_id'), header=None)

    # Loading the model (PyTorch)
    parameters = config.config(os.path.join(model_dir, config.DEFAULT_FILENAME))  # Load parameters (for model architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlasGraphModel(parameters).to(device)  # Instantiate the PyTorch model
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))  # Load the saved weights
    model.eval()  # Set to evaluation mode

    # Predict
    for idx, row in test_files.iterrows():
        graph_file = row['graph']
        sample_id = row['sample_id']
        prediction_df = test_one(file_prefix, graph_file, model, parameters, sample_id, device)  # Pass device

        if idx == 0:
            prediction_df.to_csv(output_file, header=True, index=False, mode='w')
        else:
            prediction_df.to_csv(output_file, header=False, index=False, mode='a')


def main_gfa(
    graph_file: 'gfa or gfa.gz file',
    model_dir: 'name of an input folder with the trained model',
    output_file: 'name of the output csv file'
):
    """Runs a trained model on a single gfa or gfa.gz file."""

    # Load parameters from YAML config (same as TF version)
    parameters = config.config(os.path.join(model_dir, config.DEFAULT_FILENAME))

    # Determine the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate model and load weights (PyTorch)
    model = PlasGraphModel(parameters).to(device)  # Create the PyTorch model and move it to the device
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))  # Load the weights, mapping to the device
    model.eval()  # Set the model to evaluation mode

    # predict
    prediction_df = test_one('', graph_file, model, parameters, '.', device)  # Pass the device to test_one
    # write output to a file
    prediction_df.to_csv(output_file, header=True, index=False, mode='w')


import tempfile # Import tempfile

def test_one(file_prefix, graph_file, model, parameters, sample_id, device): # Add device
    """Process a single graph file for classification with a PyTorch model."""
    
    G = create_graph.read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)  # fix order of nodes

    #  Create a temporary directory for Networkx_to_PyG
    with tempfile.TemporaryDirectory() as tmpdir:
        the_graph_dataset = create_graph.Networkx_to_PyG(root=tmpdir, nx_graph=G, node_order=node_list, parameters=parameters)
        data = the_graph_dataset[0]

    # Move data to the device
    data = data.to(device)

    # Apply GCN Filter manually (PyG)
    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    row, col = edge_index
    deg = degree(row, data.num_nodes, dtype=torch.float)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    data.edge_index = edge_index
    data.edge_weight = edge_weight
    
    # compute predictions (PyTorch)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        preds = model(data).cpu().numpy()  # Get predictions, move to CPU for NumPy

    # prediction to df (same as original)
    list_of_lists_with_prediction = []
    for index, contig_id in enumerate(node_list):
        contig_short = G.nodes[contig_id]["contig"]
        contig_len = G.nodes[contig_id]["length"]
        plasmid_probability = preds[index][0]
        chromosome_probability = preds[index][1]
        label = helpers.pair_to_label(list(np.around(preds[index])))
        list_of_lists_with_prediction.append(
            [sample_id, contig_short, contig_len, plasmid_probability, chromosome_probability, label]
        )

    prediction_df = pd.DataFrame(
        list_of_lists_with_prediction,
        columns=[
            "sample",
            "contig",
            "length",
            "plasmid_score",
            "chrom_score",
            "label",
        ],
    )

    return prediction_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=inspect.getdoc(inspect.getmodule(main_set)), formatter_class=argparse.RawDescriptionHelpFormatter)

    subparser = parser.add_subparsers(dest='command')
    cmd_set = subparser.add_parser('set', description=inspect.getdoc(main_set))
    cmd_set.add_argument("test_file_list", help="csv file with a list of testing samples")
    cmd_set.add_argument("file_prefix", help="common prefix to be used for all filenames listed in test_file_list, e.g. ../data/")
    cmd_set.add_argument("model_dir", help="name of the input folder with the trained model")
    cmd_set.add_argument("output_file", help="name of the output csv file")
    
    cmd_set = subparser.add_parser('gfa', description=inspect.getdoc(main_gfa))
    cmd_set.add_argument("graph_file", help="input gfa or gfa.gz file")
    cmd_set.add_argument("model_dir", help="name of the input folder with the trained model")
    cmd_set.add_argument("output_file", help="name of the output csv file")
    
    args = parser.parse_args()
    arg_dict = vars(args).copy()
    del arg_dict['command']
    if args.command == 'set':
        main_set(**arg_dict)
    elif args.command == 'gfa':
        main_gfa(**arg_dict)
    else:
        parser.print_usage()
                
