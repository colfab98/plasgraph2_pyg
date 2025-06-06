import inspect
import pandas as pd
import networkx as nx
import numpy as np
import os
import random
import argparse
import torch
import yaml # For loading the config

# Import your project's modules
import create_graph
import config as pl_config # Renamed to avoid conflict with this script's config
from architecture import GCNModel, GGNNModel
# Thresholds module is not directly used for applying, as it's part of model's config
# but evaluation logic is similar.

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt


def diagnose_graph_data(data, parameters):
    """
    Prints diagnostic information about the PyTorch Geometric graph data.
    Copied and adapted from your training script.
    """
    print("=" * 60)
    print("🔍 PyTorch Geometric Graph Diagnostic (Test Set)")
    print("=" * 60)
    print()

    print(f"type(data): {type(data)}")
    if hasattr(data, 'x'):
        print(f"type(data.x): {type(data.x)}")
        x = data.x
        print(f"🧬 Feature matrix shape: {x.shape}")
        print(f"📋 Features used (from config): {parameters['features']}")
        print(f"🧾 Sample features (first 2 nodes):\n{x[:2]}")
    else:
        print("⚠️ data.x not found.")

    if hasattr(data, 'y'):
        print(f"type(data.y): {type(data.y)}")
        y = data.y
        print(f"\n🏷️ Label matrix shape: {y.shape}")
        print(f"🔢 Label sample (first 5):\n{y[:5]}")

        y_np = y.detach().cpu().numpy()
        unique_labels, counts = np.unique(y_np, axis=0, return_counts=True)
        print("\n📊 Label counts:")
        for label_val, count in zip(unique_labels, counts):
            print(f"  {label_val.tolist()} → {count} nodes")
    else:
        print("⚠️ data.y not found.")

    if hasattr(data, 'edge_index'):
        print(f"type(data.edge_index): {type(data.edge_index)}")
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        num_nodes = data.num_nodes
        print(f"\n🔗 Edge index shape: {edge_index.shape}")
        print(f"  → Number of edges: {num_edges}")
        density = num_edges / (num_nodes ** 2) if num_nodes > 0 else 0
        print(f"  → Density: {density:.6f}")

        degrees = torch.bincount(edge_index[0], minlength=num_nodes) if num_nodes > 0 else torch.tensor([])
        num_isolated = torch.sum(degrees == 0).item() if num_nodes > 0 else 0
        print(f"🧍 Isolated nodes: {num_isolated}")
    else:
        print("⚠️ data.edge_index not found.")

    print(f"\n📦 Available attributes in data object: {list(data.keys()) if hasattr(data, 'keys') else 'N/A'}")
    print("=" * 60)


def main(best_config_file: 'YAML configuration file of the best model saved',
         trained_model_dir: 'Directory containing the trained model and its config',
         test_file_list: 'CSV file with a list of test samples (e.g., eskapee-test.csv)',
         file_prefix: 'Common prefix for filenames in test_file_list (e.g., plasgraph2-datasets/)',
         log_dir: 'Optional output folder for test logs or plots' = None):
    
    best_config_path = os.path.join(trained_model_dir, best_config_file)

    best_parameters = pl_config.config(best_config_path)
    best_parameters.config_file_path = best_config_path

    test_data_root = os.path.join(trained_model_dir, "test_data_processed")
    os.makedirs(test_data_root, exist_ok=True)


    all_test_graphs = create_graph.Dataset_Pytorch(
        root = test_data_root, 
        file_prefix = file_prefix,
        train_file_list=test_file_list,
        parameters = best_parameters 
    )

    data_test = all_test_graphs[0]
    G_test = all_test_graphs.G 
    node_list_test = all_test_graphs.node_list 

    diagnose_graph_data(data_test, best_parameters)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    node_to_idx_map = {node_id: i for i, node_id in enumerate(node_list_test)}

    num_total_nodes_test = data_test.num_nodes
    labels_text_test = [G_test.nodes[node_id]["text_label"] for node_id in node_list_test]
    num_chromosome_test = labels_text_test.count("chromosome")
    num_plasmid_test = labels_text_test.count("plasmid")
    num_ambiguous_test = labels_text_test.count("ambiguous")
    num_unlabeled_test = labels_text_test.count("unlabeled")

    assert num_total_nodes_test == num_chromosome_test + num_plasmid_test + num_ambiguous_test + num_unlabeled_test

    print(labels_text_test[:50])


    print("Chromosome contigs:", num_chromosome_test,
          "Plasmid contigs:", num_plasmid_test,
          "Ambiguous contigs:", num_ambiguous_test,
          "Unlabeled contigs:", num_unlabeled_test)
    
    chromosome_weight = 1.0
    plasmid_weight = float(best_parameters["plasmid_ambiguous_weight"])
    ambiguous_weight = plasmid_weight

    masks_test_values = []
    for node_id in node_list_test:
        label = G_test.nodes[node_id]["text_label"]
        if label == "unlabeled":
            masks_test_values.append(0.0)
        elif label == "chromosome":
            masks_test_values.append(chromosome_weight)
        elif label == "plasmid":
            masks_test_values.append(plasmid_weight)
        elif label == "ambiguous":
            masks_test_values.append(ambiguous_weight)
    masks_test = torch.tensor(masks_test_values, dtype=torch.float32)


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_test = data_test.to(device)
    masks_test = masks_test.to(device)

    # Initialize model
    if best_parameters['model_type'] == 'GCNModel':
        model = GCNModel(best_parameters).to(device)
    elif best_parameters['model_type'] == 'GGNNModel':
        model = GGNNModel(best_parameters).to(device)
    else:
        raise ValueError(f"Unsupported model type in best trial's config: {best_parameters['model_type']}")

    model_weights_path = os.path.join(trained_model_dir, "final_retrained_best_model.pt")

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        all_nodes_outputs = model(data_test)
    
    sample_ids = sorted(list(set(G_test.nodes[node_id]["sample"] for node_id in node_list_test)))

    all_f1_plasmid_scores = []
    all_acc_plasmid_scores = []
    all_prec_plasmid_scores = []
    all_rec_plasmid_scores = []
    all_auroc_plasmid_scores = []

    all_f1_chromosome_scores = []
    all_acc_chromosome_scores = []
    all_prec_chromosome_scores = []
    all_rec_chromosome_scores = []
    all_auroc_chromosome_scores = []


    for sample_id in sample_ids:
        current_sample_node_indices_global = [
            node_to_idx_map[node_id] 
            for node_id in node_list_test 
            if G_test.nodes[node_id]["sample"] == sample_id
        ]

        current_sample_node_indices_global = torch.tensor(current_sample_node_indices_global, dtype=torch.long, device=device)

        y_sample = data_test.y[current_sample_node_indices_global]
        outputs_sample = all_nodes_outputs[current_sample_node_indices_global]
        masks_sample = masks_test[current_sample_node_indices_global]

        sample_node_ids = [nid for nid in node_list_test if G_test.nodes[nid]["sample"] == sample_id]
        labels_text_sample = [G_test.nodes[node_id]["text_label"] for node_id in sample_node_ids]
        num_total_sample = len(labels_text_sample)
        num_chromosome_sample = labels_text_sample.count("chromosome")
        num_plasmid_sample = labels_text_sample.count("plasmid")
        num_ambiguous_sample = labels_text_sample.count("ambiguous")
        num_unlabeled_sample = labels_text_sample.count("unlabeled")

        active_mask_for_sample = masks_sample > 0

        probs_plasmid_sample = outputs_sample[:, 0]
        probs_chromosome_sample = outputs_sample[:, 1]

        # --- Plasmid Evaluation for the current sample ---
        labels_plasmid_sample_active = y_sample[active_mask_for_sample, 0]
        probs_plasmid_sample_active = probs_plasmid_sample[active_mask_for_sample]
        
        y_true_plasmid = labels_plasmid_sample_active.cpu().numpy()
        y_probs_plasmid = probs_plasmid_sample_active.cpu().numpy()
        sample_weight_plasmid = masks_sample[active_mask_for_sample].cpu().numpy()
        
        final_best_thresh_plasmid = best_parameters['plasmid_threshold'] #
        y_pred_plasmid = (y_probs_plasmid >= final_best_thresh_plasmid).astype(int) #

        acc_plasmid = accuracy_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid) #
        prec_plasmid = precision_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0) #
        rec_plasmid = recall_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0) #
        f1_plasmid = f1_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0) #    
        if len(np.unique(y_true_plasmid)) < 2:
            auroc_plasmid = np.nan
        else:
            auroc_plasmid = roc_auc_score(y_true_plasmid, y_probs_plasmid, sample_weight=sample_weight_plasmid)  
            all_auroc_plasmid_scores.append(auroc_plasmid) 

        print(f"  Sample {sample_id} - Test PLASMID Metrics | F1: {f1_plasmid:.4f} @ Thresh: {final_best_thresh_plasmid:.2f} | Acc: {acc_plasmid:.4f} | Prec: {prec_plasmid:.4f} | Rec: {rec_plasmid:.4f} | AUROC: {auroc_plasmid:.4f}") #
        
        all_f1_plasmid_scores.append(f1_plasmid)
        all_acc_plasmid_scores.append(acc_plasmid)
        all_prec_plasmid_scores.append(prec_plasmid)
        all_rec_plasmid_scores.append(rec_plasmid)       


        # --- Chromosome Evaluation for the current sample ---
        labels_chromosome_sample_active = y_sample[active_mask_for_sample, 1]
        probs_chromosome_sample_active = probs_chromosome_sample[active_mask_for_sample]

        y_true_chromosome = labels_chromosome_sample_active.cpu().numpy()
        y_probs_chromosome = probs_chromosome_sample_active.cpu().numpy()
        sample_weight_chromosome = masks_sample[active_mask_for_sample].cpu().numpy()

        final_best_thresh_chromosome = best_parameters['chromosome_threshold'] #
        y_pred_chromosome = (y_probs_chromosome >= final_best_thresh_chromosome).astype(int) #

        acc_chromosome = accuracy_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome) #
        prec_chromosome = precision_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0) #
        rec_chromosome = recall_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0) #
        f1_chromosome = f1_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0) #
        auroc_chromosome = roc_auc_score(y_true_chromosome, y_probs_chromosome, sample_weight=sample_weight_chromosome) #

        if len(np.unique(y_true_plasmid)) < 2:
            auroc_chromosome = np.nan
        else:
            auroc_chromosome = roc_auc_score(y_true_chromosome, y_probs_chromosome, sample_weight=sample_weight_chromosome)  
            all_auroc_chromosome_scores.append(auroc_chromosome)

        print(f"  Sample {sample_id} - Test CHROMOSOME Metrics | F1: {f1_chromosome:.4f} @ Thresh: {final_best_thresh_chromosome:.2f} | Acc: {acc_chromosome:.4f} | Prec: {prec_chromosome:.4f} | Rec: {rec_chromosome:.4f} | AUROC: {auroc_chromosome:.4f}") #
        
        all_f1_chromosome_scores.append(f1_chromosome)
        all_acc_chromosome_scores.append(acc_chromosome)
        all_prec_chromosome_scores.append(prec_chromosome)
        all_rec_chromosome_scores.append(rec_chromosome)


    median_f1_plasmid = np.median(all_f1_plasmid_scores)
    median_acc_plasmid = np.median(all_acc_plasmid_scores)
    median_prec_plasmid = np.median(all_prec_plasmid_scores)
    median_rec_plasmid = np.median(all_rec_plasmid_scores)
    median_auroc_plasmid = np.median(all_auroc_plasmid_scores)
    print(f"Median PLASMID Metrics | F1: {median_f1_plasmid:.4f} | Acc: {median_acc_plasmid:.4f} | Prec: {median_prec_plasmid:.4f} | Rec: {median_rec_plasmid:.4f} | AUROC: {median_auroc_plasmid:.4f}")

    median_f1_chromosome = np.median(all_f1_chromosome_scores)
    median_acc_chromosome = np.median(all_acc_chromosome_scores)
    median_prec_chromosome = np.median(all_prec_chromosome_scores)
    median_rec_chromosome = np.median(all_rec_chromosome_scores)
    median_auroc_chromosome = np.median(all_auroc_chromosome_scores)
    print(f"Median CHROMOSOME Metrics | F1: {median_f1_chromosome:.4f} | Acc: {median_acc_chromosome:.4f} | Prec: {median_prec_chromosome:.4f} | Rec: {median_rec_chromosome:.4f} | AUROC: {median_auroc_chromosome:.4f}")



    plt.figure(figsize=(10, 7)) 
    data_to_plot = []
    labels = []
    data_to_plot.append(all_f1_plasmid_scores)
    labels.append('Plasmid F1 Scores')
    data_to_plot.append(all_f1_chromosome_scores)
    labels.append('Chromosome F1 Scores')
    parts = plt.violinplot(data_to_plot, showmeans=False, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor('skyblue' if labels[i].startswith('Plasmid') else 'lightcoral')
                pc.set_edgecolor('black')
                pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('grey')
            vp.set_linewidth(1)
    parts['cmedians'].set_edgecolor('red')
    parts['cmedians'].set_linewidth(1.5)
    for i, d_list in enumerate(data_to_plot): 
            x_jitter = np.random.normal(loc=i + 1, scale=0.04, size=len(d_list))
            plt.scatter(x_jitter, d_list, alpha=0.4, s=20, color='dimgray', zorder=3) # zorder to plot dots on top
    plt.ylabel('F1 Score', fontsize=14)
    plt.title('Distribution of F1 Scores Across Samples', fontsize=16)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=0, ha='center', fontsize=12) # Use np.arange
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y') # Grid for y-axis
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plot_filename = "f1_scores_violin_plot.png"
    plot_path = os.path.join(log_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained plASgraph model on a test dataset.")
    parser.add_argument("best_config_file", help="YAML configuration file of the best model saved")
    parser.add_argument("trained_model_dir", help="Directory containing the trained model (final_retrained_best_model.pt) and its configuration file (e.g., best_config_with_thresholds.yaml). This is the output directory from plASgraph2_train16.py.")
    parser.add_argument("test_file_list", help="CSV file listing test samples (e.g., path/to/plasgraph2-datasets/eskapee-test.csv).")
    parser.add_argument("file_prefix", help="Common prefix for all filenames in test_file_list (e.g., path/to/plasgraph2-datasets/). Ensure this points to the directory containing the GFA files listed in test_file_list.")
    parser.add_argument("-l", "--log_dir", dest="log_dir", help="Optional output folder for test logs or plots", default=None)
    
    args = parser.parse_args()
    main(**vars(args))