import gzip
import os
import pandas as pd
import networkx as nx
import numpy as np
import argparse
import inspect
import architecture

import torch_geometric


import create_graph as create_graph
import helpers
import config
import torch



def main_set(
    test_file_list: 'csv file with a list of testing samples',
    file_prefix: 'common prefix to be used for all filenames listed in test_file_list, e.g. ../data/',
    model_dir: 'name of an input folder with the trained model',
    output_file: 'name of the output csv file'
):
    test_files = pd.read_csv(test_file_list, names=('graph', 'csv', 'sample_id'), header=None)
    config_path = os.path.join(model_dir, "best_config_with_thresholds.yaml")
    parameters = config.config(config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if parameters['model_type'] == 'GCNModel':
        model = architecture.GCNModel(parameters).to(device)
    elif parameters['model_type'] == 'GGNNModel':
        model = architecture.GGNNModel(parameters).to(device)
    else:
        raise ValueError(f"Unsupported model type in config: {parameters['model_type']}")
    
    weights_path = os.path.join(model_dir, "final_retrained_best_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()

    for idx, row in test_files.iterrows():
        graph_file = row['graph']
        sample_id = row['sample_id']
        prediction_df = test_one(file_prefix, graph_file, model, parameters, sample_id, device)

        if idx == 0:
            prediction_df.to_csv(output_file, header=True, index=False, mode='w')
        else:
            prediction_df.to_csv(output_file, header=False, index=False, mode='a')



def main_gfa(
    graph_file: 'gfa or gfa.gz file',
    model_dir: 'name of an input folder with the trained model',
    output_file: 'name of the output csv file'
):
   
    config_path = os.path.join(model_dir, "best_config_with_thresholds.yaml")
    parameters = config.config(config_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if parameters['model_type'] == 'GCNModel':
        model = architecture.GCNModel(parameters).to(device)
    elif parameters['model_type'] == 'GGNNModel':
        model = architecture.GGNNModel(parameters).to(device)
    else:
        raise ValueError(f"Unsupported model type in config: {parameters['model_type']}")
    
    weights_path = os.path.join(model_dir, "final_retrained_best_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()

    sample_id = os.path.basename(graph_file).split('.')[0]
    prediction_df = test_one(
        file_prefix='', 
        graph_file=graph_file, 
        model=model, 
        parameters=parameters, 
        sample_id=sample_id, 
        device=device
    )

    prediction_df.to_csv(output_file, header=True, index=False, mode='w')




def test_one(file_prefix: str, graph_file: str, model: torch.nn.Module, parameters: object, sample_id: str, device: torch.device):
    
    G = create_graph.read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)

    for u, v in G.edges():
        kmer_u = np.array(G.nodes[u]["kmer_counts_norm"])
        kmer_v = np.array(G.nodes[v]["kmer_counts_norm"])
        dot_product = np.dot(kmer_u, kmer_v)
        G.edges[u, v]["kmer_dot_product"] = dot_product

    features = parameters["features"]
    x = np.array([[G.nodes[node_id][f] for f in features] for node_id in node_list])
    x_tensor = torch.tensor(x, dtype=torch.float)

    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}

    edge_sources = []
    edge_targets = []
    edge_attrs = []
    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_sources.extend([u_idx, v_idx])
        edge_targets.extend([v_idx, u_idx])
        dot_product = data.get("kmer_dot_product", 0.0)
        edge_attrs.extend([[dot_product], [dot_product]])
    
    edge_index_tensor = torch.tensor(np.vstack((edge_sources, edge_targets)), dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)

    data = torch_geometric.data.Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor
    )
    preds = architecture.apply_to_graph(model, data, parameters)

    output_rows = []
    for i, node_id in enumerate(node_list):
        contig_short = G.nodes[node_id]["contig"]
        contig_len = G.nodes[node_id]["length"]
        
        plasmid_score = preds[i, 0]
        chrom_score = preds[i, 1]
        label = helpers.pair_to_label(list(np.round(preds[i])))

        output_rows.append(
            [sample_id, contig_short, contig_len, plasmid_score, chrom_score, label]
        )

    prediction_df = pd.DataFrame(
        output_rows,
        columns=["sample", "contig", "length", "plasmid_score", "chrom_score", "label"]
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
                
