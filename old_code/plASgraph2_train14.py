#!/usr/bin/env python3
# coding: utf-8

import inspect
import pandas as pd
import networkx as nx
import numpy as np
import os
import random
import argparse

import torch

import create_graph as create_graph
import config # Assuming config.py defines a 'config' class
import thresholds

from architecture import GCNModel, GGNNModel # Assuming these are defined elsewhere

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 

import matplotlib.pyplot as plt
import optuna # Optuna import

def fix_gradients(config_params, model: torch.nn.Module): # Name changed back
    """
    Cleans gradients by removing NaN/Inf and optionally clips them.
    Uses config_params which should be an instance of config.config
    """
    # Remove NaN and Inf from gradients
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)

    # Clip gradients before optimization step
    # Access gradient_clipping value from the config object's _params attribute, matching original
    gradient_clipping_value = config_params._params.get('gradient_clipping', 0.0) # Changed access method
                                                                        
    if gradient_clipping_value is not None and gradient_clipping_value > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)

# Optuna Objective Function
# Parameter names changed to match what's passed from main: parameters, model_output_dir, log_dir, G
def objective(trial, parameters, data, masks_train, masks_validate, device, model_output_dir, log_dir, node_list, G):
    """
    Optuna objective function. Trains the model with hyperparameters suggested by Optuna.
    """
    # Create a dictionary of parameters for this trial, starting from a copy of base parameters
    # Accesses _params on the 'parameters' object (which is base_parameters_obj)
    trial_params_dict = parameters._params.copy() 

    # Suggest hyperparameters
    # Ranges for these suggestions can also be part of the base_parameters_obj for more flexibility
    trial_params_dict['learning_rate'] = trial.suggest_float(
        "learning_rate", 
        parameters._params.get('learning_rate_min', 1e-4), 
        parameters._params.get('learning_rate_max', 1e-2), 
        log=True
    )
    trial_params_dict['l2_reg'] = trial.suggest_float(
        "l2_reg", 
        parameters._params.get('l2_reg_min', 1e-6), 
        parameters._params.get('l2_reg_max', 1e-3), 
        log=True
    )
    trial_params_dict['n_channels'] = trial.suggest_int(
        "n_channels", 
        parameters._params.get('n_channels_min', 16), # Consider if these min/max keys also need renaming
        parameters._params.get('n_channels_max', 128), # or if the values are fine as hardcoded defaults
        step=parameters._params.get('n_channels_step', 16)
    )
    trial_params_dict['n_gnn_layers'] = trial.suggest_int(
        "n_gnn_layers", 
        parameters._params.get('n_gnn_layers_min', 1), # Consider if these min/max keys also need renaming
        parameters._params.get('n_gnn_layers_max', 5)
)
    trial_params_dict['dropout_rate'] = trial.suggest_float(
        "dropout_rate", 
        parameters._params.get('dropout_rate_min', 0.0), 
        parameters._params.get('dropout_rate_max', 0.7)
    )
    trial_params_dict['gradient_clipping'] = trial.suggest_float(
        "gradient_clipping", 
        parameters._params.get('gradient_clipping_min', 0.0), 
        parameters._params.get('gradient_clipping_max', 1.0)
    )

    # Create a new config object for this specific trial.
    # Assumes config.config can be initialized from a file path (stored in parameters.config_file_path)
    # and then its internal _params dictionary can be overwritten.
    # This part might need adjustment based on your actual 'config.py' implementation.
    # The try-except block was removed as per user request to avoid adding unnecessary try-excepts.
    trial_config_obj = config.config(parameters.config_file_path) # Re-init from original file path stored in the main 'parameters' object
    trial_config_obj._params = trial_params_dict # Directly overwrite internal params with trial-specific ones


    trials_model_base_dir = os.path.join(model_output_dir, "trials")
    os.makedirs(trials_model_base_dir, exist_ok=True) # Create 'trials' folder if it doesn't exist

    # Define the specific directory for this trial's model outputs
    model_output_dir_trial = os.path.join(trials_model_base_dir, f"trial_{trial.number}")
    os.makedirs(model_output_dir_trial, exist_ok=True) # Create specific trial folder
    
    log_dir_trial = None
    if log_dir: 
        # Define the base directory for all trial logs within log_dir
        trials_log_base_dir = os.path.join(log_dir, "trials")
        os.makedirs(trials_log_base_dir, exist_ok=True) # Create 'trials' folder if it doesn't exist

        # Define the specific directory for this trial's logs
        log_dir_trial = os.path.join(trials_log_base_dir, f"trial_{trial.number}")
        os.makedirs(log_dir_trial, exist_ok=True) # Create specific trial log folder

    # --- Training and Evaluation Logic (adapted from main) ---
    # Accesses parameters from trial_config_obj using dictionary-style access,
    # consistent with how 'parameters' was used in the original main script.
    if trial_config_obj['model_type'] == 'GCNModel':
        model = GCNModel(trial_config_obj).to(device)
    elif trial_config_obj['model_type'] == 'GGNNModel':
        model = GGNNModel(trial_config_obj).to(device)
    else:
        raise ValueError(f"Unsupported model type in trial {trial.number}: {trial_config_obj['model_type']}")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trial_config_obj['learning_rate'],
        weight_decay=trial_config_obj['l2_reg']
    )

    if trial_config_obj["loss_function"] == "crossentropy":
        criterion = torch.nn.BCELoss(reduction='none')
    elif trial_config_obj["loss_function"] == "mse":
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss function in trial {trial.number}: {trial_config_obj['loss_function']}")

    best_val_loss_for_trial = float("inf")
    patience_for_trial = 0
    
    train_losses_trial = [] 
    val_losses_trial = []   

    for epoch in range(trial_config_obj["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(data) 
        
        valid_label_mask = data.y.sum(dim=1) != 0
        masked_outputs_train = outputs[valid_label_mask]
        masked_labels_train = data.y[valid_label_mask]
        masked_weights_train = masks_train[valid_label_mask] 

        loss_per_node_train = criterion(masked_outputs_train, masked_labels_train)
        train_loss = (loss_per_node_train.sum(dim=1) * masked_weights_train).sum() / masked_weights_train.sum()

        train_loss.backward()
        fix_gradients(trial_config_obj, model) # Pass the config object for this trial
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(data)
            masked_outputs_val = val_outputs[valid_label_mask]
            masked_labels_val = data.y[valid_label_mask]
            masked_weights_val = masks_validate[valid_label_mask] 

            loss_per_node_val = criterion(masked_outputs_val, masked_labels_val)
            val_loss = (loss_per_node_val.sum(dim=1) * masked_weights_val).sum() / masked_weights_val.sum()
        
        train_losses_trial.append(train_loss.item()) 
        val_losses_trial.append(val_loss.item())     

        if val_loss.item() < best_val_loss_for_trial: 
            best_val_loss_for_trial = val_loss.item()
            patience_for_trial = 0
            torch.save(model.state_dict(), os.path.join(model_output_dir_trial, "best_model.pt"))
            trial_config_obj.write_yaml(os.path.join(model_output_dir_trial, config.DEFAULT_FILENAME))
        else:
            patience_for_trial += 1
            if patience_for_trial >= trial_config_obj["early_stopping_patience"]:
                break
    
    if train_losses_trial and val_losses_trial : 
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_trial, label='Train Loss')
        plt.plot(val_losses_trial, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Trial {trial.number} - Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(1, max(train_losses_trial) if train_losses_trial else 1, max(val_losses_trial) if val_losses_trial else 1) * 1.1) 
        plt.savefig(os.path.join(model_output_dir_trial, "loss_over_epochs.png"))
        plt.close()

    print(f"Trial {trial.number} finished. Best Val Loss for trial: {best_val_loss_for_trial:.4f}. Params: {trial.params}")
    
    return best_val_loss_for_trial


def main(config_file: 'YAML configuration file',
         train_file_list: 'csv file with a list of training samples',
         file_prefix: 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
         model_output_dir: 'name of an output folder with the trained model',
         gml_file: 'optional output file for the graph in GML format' = None,
         log_dir: 'optional output folder for various training logs' = None):

    parameters = config.config(config_file)
    # Store the original config file path on the parameters object itself,
    # so it can be accessed within the objective function if needed for re-initialization.
    parameters.config_file_path = config_file 

    all_graphs = create_graph.Dataset_Pytorch(
        root = model_output_dir, 
        file_prefix = file_prefix,
        train_file_list = train_file_list,
        parameters = parameters 
    )

    data = all_graphs[0]
    G = all_graphs.G # This G will be passed to objective
    node_list = all_graphs.node_list # This node_list will be passed to objective

    os.makedirs(model_output_dir, exist_ok=True)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    number_total_nodes = data.num_nodes
    # Using dictionary-style access for 'text_label', assuming G.nodes[node_id] is a dict or dict-like
    labels_text = [G.nodes[node_id]["text_label"] for node_id in node_list]  
    num_chromosome = labels_text.count("chromosome")
    num_plasmid = labels_text.count("plasmid")
    num_ambiguous = labels_text.count("ambiguous")
    num_unlabeled = labels_text.count("unlabeled")

    print("Chromosome contigs:", num_chromosome,
          "Plasmid contigs:", num_plasmid,
          "Ambiguous contigs:", num_ambiguous,
          "Unlabeled contigs:", num_unlabeled)

    # Accessing parameters using dictionary-style access, as in original main
    chromosome_weight = 1.0 
    plasmid_weight = float(parameters["plasmid_ambiguous_weight"])
    ambiguous_weight = plasmid_weight

    masks_list = [] 
    for node_id in node_list:
        label_text_node = G.nodes[node_id]["text_label"] 
        if label_text_node == "unlabeled":
            masks_list.append(0.0)
        elif label_text_node == "chromosome":
            masks_list.append(chromosome_weight)
        elif label_text_node == "plasmid":
            masks_list.append(plasmid_weight)
        elif label_text_node == "ambiguous":
            masks_list.append(ambiguous_weight)
    masks_np = np.array(masks_list, dtype=float) 

    seed_number = parameters["random_seed"] # Dictionary-style access
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number + 1)
    torch.manual_seed(seed_number + 2)
    
    train_indices = np.random.rand(len(masks_np)) <= 0.8 
    
    masks_train_np_final = masks_np.copy()
    masks_validate_np_final = masks_np.copy()

    masks_train_np_final[~train_indices] = 0.0 
    masks_validate_np_final[train_indices] = 0.0 

    masks_train = torch.tensor(masks_train_np_final, dtype=torch.float32)
    masks_validate = torch.tensor(masks_validate_np_final, dtype=torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    masks_train = masks_train.to(device)
    masks_validate = masks_validate.to(device)
    
    # Accessing parameters using _params.get for Optuna-specific settings, providing defaults
    study_name = parameters._params.get("optuna_study_name", "plASgraph_optimization")
    n_trials = parameters._params.get("optuna_n_trials", 20) 

    study = optuna.create_study(direction="minimize")

    # Pass 'parameters', 'model_output_dir', 'log_dir', 'node_list', 'G' to objective
    study.optimize(lambda trial: objective(trial, parameters, data, masks_train, masks_validate, device, model_output_dir, log_dir, node_list, G), 
                   n_trials=n_trials)

    print("\nOptuna study finished.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Validation Loss): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_model_output_dir = os.path.join(model_output_dir, f"trial_{best_trial.number}")
    best_model_path = os.path.join(best_model_output_dir, "best_model.pt")
    best_config_path = os.path.join(best_model_output_dir, config.DEFAULT_FILENAME)

    print(f"\nLoading best model from: {best_model_path}")
    print(f"Loading best config from: {best_config_path}")

    best_parameters = config.config(best_config_path) 

    # Dictionary-style access for best_parameters
    if best_parameters['model_type'] == 'GCNModel':
        model = GCNModel(best_parameters).to(device)
    elif best_parameters['model_type'] == 'GGNNModel':
        model = GGNNModel(best_parameters).to(device)
    else:
        raise ValueError(f"Unsupported model type in best trial: {best_parameters['model_type']}")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if best_parameters['set_thresholds']: # Dictionary-style access
        log_dir_best_trial = None
        if log_dir:
            log_dir_best_trial = os.path.join(log_dir, f"trial_{best_trial.number}")
            os.makedirs(log_dir_best_trial, exist_ok=True) 
        
        print(f"Setting thresholds using best model. Log directory for thresholds: {log_dir_best_trial if log_dir_best_trial else 'None'}")
        thresholds.set_thresholds(model, data, masks_validate, best_parameters, log_dir_best_trial) 
        
        # Define a consistent name for the config file with thresholds
        config_with_thresholds_filename = "best_config_with_thresholds.yaml"
        if hasattr(config, 'DEFAULT_FILENAME_WITH_THRESHOLDS'):
            config_with_thresholds_filename = config.DEFAULT_FILENAME_WITH_THRESHOLDS
        
        best_parameters.write_yaml(os.path.join(best_model_output_dir, config_with_thresholds_filename))
        print(f"Optimal thresholds saved to config file in {best_model_output_dir} as {config_with_thresholds_filename}")


    print("\n--- Final Evaluation Metrics on Validation Set (Best Optuna Trial) ---")
    model.eval() 
    with torch.no_grad():
        final_val_outputs = model(data) 

        probs_plasmid_final = final_val_outputs[:, 0]
        probs_chromosome_final = final_val_outputs[:, 1]

        active_val_mask = masks_validate > 0 
        
        labels_plasmid_val_active = data.y[active_val_mask, 0]
        probs_plasmid_val_active = probs_plasmid_final[active_val_mask]
        valid_eval_indices_plasmid = labels_plasmid_val_active != -1 
        
        y_true_plasmid = labels_plasmid_val_active[valid_eval_indices_plasmid].cpu().numpy()
        y_probs_plasmid = probs_plasmid_val_active[valid_eval_indices_plasmid].cpu().numpy()
        sample_weight_plasmid = masks_validate[active_val_mask][valid_eval_indices_plasmid].cpu().numpy()

        # Dictionary-style access for best_parameters
        final_best_thresh_plasmid = best_parameters['plasmid_threshold']
        y_pred_plasmid = (y_probs_plasmid >= final_best_thresh_plasmid).astype(int)

        if len(y_true_plasmid) > 0: 
            acc_plasmid = accuracy_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid)
            prec_plasmid = precision_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            rec_plasmid = recall_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0)
            f1_plasmid = f1_score(y_true_plasmid, y_pred_plasmid, sample_weight=sample_weight_plasmid, zero_division=0) 
            auroc_plasmid = roc_auc_score(y_true_plasmid, y_probs_plasmid, sample_weight=sample_weight_plasmid)
            print(f"Final PLASMID Metrics | F1: {f1_plasmid:.4f} @ Thresh: {final_best_thresh_plasmid:.2f} | Acc: {acc_plasmid:.4f} | Prec: {prec_plasmid:.4f} | Rec: {rec_plasmid:.4f} | AUROC: {auroc_plasmid:.4f}")
        else:
            print("No valid samples for PLASMID final evaluation.")

        labels_chromosome_val_active = data.y[active_val_mask, 1]
        probs_chromosome_val_active = probs_chromosome_final[active_val_mask]
        valid_eval_indices_chromosome = labels_chromosome_val_active != -1

        y_true_chromosome = labels_chromosome_val_active[valid_eval_indices_chromosome].cpu().numpy()
        y_probs_chromosome = probs_chromosome_val_active[valid_eval_indices_chromosome].cpu().numpy()
        sample_weight_chromosome = masks_validate[active_val_mask][valid_eval_indices_chromosome].cpu().numpy()

        # Dictionary-style access for best_parameters
        final_best_thresh_chromosome = best_parameters['chromosome_threshold']
        y_pred_chromosome = (y_probs_chromosome >= final_best_thresh_chromosome).astype(int)

        if len(y_true_chromosome) > 0:
            acc_chromosome = accuracy_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome)
            prec_chromosome = precision_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            rec_chromosome = recall_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0)
            f1_chromosome = f1_score(y_true_chromosome, y_pred_chromosome, sample_weight=sample_weight_chromosome, zero_division=0) 
            auroc_chromosome = roc_auc_score(y_true_chromosome, y_probs_chromosome, sample_weight=sample_weight_chromosome)
            print(f"Final CHROMOSOME Metrics | F1: {f1_chromosome:.4f} @ Thresh: {final_best_thresh_chromosome:.2f} | Acc: {acc_chromosome:.4f} | Prec: {prec_chromosome:.4f} | Rec: {rec_chromosome:.4f} | AUROC: {auroc_chromosome:.4f}")
        else:
            print("No valid samples for CHROMOSOME final evaluation.")

    if gml_file and G: 
        # nx.write_gml(G, gml_file) # Commented out as in previous version
        print(f"Graph GML file saving skipped in this Optuna version (was: {gml_file})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=inspect.getdoc(main))
    parser.add_argument("config_file", help="YAML configuration file")
    parser.add_argument("train_file_list", help="CSV file listing training samples")
    parser.add_argument("file_prefix", help="Common prefix for all filenames in train_file_list, e.g. ../data/")
    parser.add_argument("model_output_dir", help="Name of the base output folder for the trained models and Optuna trials")
    parser.add_argument("-g", "--gml_file", dest="gml_file", help="Optional output GML graph file", default=None) 
    parser.add_argument("-l", "--log_dir", dest="log_dir", help="Optional base output folder for training logs", default=None) 
    args = parser.parse_args()
    main(**vars(args))
