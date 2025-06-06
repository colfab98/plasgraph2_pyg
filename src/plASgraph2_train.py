
import inspect
import pandas as pd
import networkx as nx
import numpy as np
import os
import random
import argparse

import torch

import create_graph as create_graph
import config 
import thresholds

from architecture import GCNModel, GGNNModel 

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score 

import matplotlib.pyplot as plt
import optuna 
import optuna.visualization as vis      

def fix_gradients(config_params, model: torch.nn.Module): 

    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad.data).any() or torch.isinf(param.grad.data).any():
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)

    gradient_clipping_value = config_params._params.get('gradient_clipping', 0.0) 
                                                                        
    if gradient_clipping_value is not None and gradient_clipping_value > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clipping_value)


def objective(trial, parameters, data, masks_train, masks_validate, device, model_output_dir, log_dir, node_list, G):
    trial_params_dict = parameters._params.copy() 

    trial_params_dict['learning_rate'] = trial.suggest_float(
        "learning_rate", 
        parameters._params.get('learning_rate_min', 1e-5), 
        parameters._params.get('learning_rate_max', 1e-3), 
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
        parameters._params.get('n_channels_min', 16), 
        parameters._params.get('n_channels_max', 128), 
        step=parameters._params.get('n_channels_step', 16)
    )
    trial_params_dict['n_gnn_layers'] = trial.suggest_int(
        "n_gnn_layers", 
        parameters._params.get('n_gnn_layers_min', 1), 
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
    trial_params_dict['edge_gate_hidden_dim'] = trial.suggest_int(
        "edge_gate_hidden_dim", 
        parameters._params.get('edge_gate_hidden_dim_min', 8), 
        parameters._params.get('edge_gate_hidden_dim_max', 64),
        step=parameters._params.get('edge_gate_hidden_dim_step', 8)
    )

    trial_config_obj = config.config(parameters.config_file_path)
    trial_config_obj._params = trial_params_dict 

    trials_model_base_dir = os.path.join(model_output_dir, "trials")
    os.makedirs(trials_model_base_dir, exist_ok=True) 

    model_output_dir_trial = os.path.join(trials_model_base_dir, f"trial_{trial.number}")
    os.makedirs(model_output_dir_trial, exist_ok=True) 
    
    log_dir_trial = None
    if log_dir: 
        trials_log_base_dir = os.path.join(log_dir, "trials")
        os.makedirs(trials_log_base_dir, exist_ok=True) 

        log_dir_trial = os.path.join(trials_log_base_dir, f"trial_{trial.number}")
        os.makedirs(log_dir_trial, exist_ok=True) 

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

    for epoch in range(trial_config_obj["epochs_trials"]):
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
        fix_gradients(trial_config_obj, model) 
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
    parameters.config_file_path = config_file 

    all_graphs = create_graph.Dataset_Pytorch(
        root = model_output_dir, 
        file_prefix = file_prefix,
        train_file_list = train_file_list,
        parameters = parameters 
    )

    data = all_graphs[0]
    G = all_graphs.G 
    node_list = all_graphs.node_list 

    def diagnose_graph(data, parameters):
        print("=" * 60)
        print("🔍 PyTorch Geometric Graph Diagnostic")
        print("=" * 60)
        print()

        # Types
        print(f"type(data): {type(data)}")  # Should be torch_geometric.data.data.Data
        print(f"type(data.x): {type(data.x)}")  # Node features → torch.Tensor
        print(f"type(data.y): {type(data.y)}")  # Labels → torch.Tensor
        print(f"type(data.edge_index): {type(data.edge_index)}")  # Edge list

        print()

        # Node features
        x = data.x
        print(f"🧬 Feature matrix shape: {x.shape}")
        print(f"📋 Features used: {parameters['features']}")
        print(f"🧾 Sample features (first 2 nodes):\n{x[:2]}")

        # Labels
        y = data.y
        print(f"\n🏷️ Label matrix shape: {y.shape}")
        print(f"🔢 Label sample (first 5):\n{y[:5]}")

        # Label distribution
        y_np = y.detach().cpu().numpy()
        unique_labels, counts = np.unique(y_np, axis=0, return_counts=True)
        print("\n📊 Label counts:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label.tolist()} → {count} nodes")

        # Edges
        edge_index = data.edge_index
        num_edges = edge_index.shape[1]
        num_nodes = data.num_nodes
        print(f"\n🔗 Edge index shape: {edge_index.shape}")
        print(f"  → Number of edges: {num_edges}")

        # Density
        density = num_edges / (num_nodes ** 2)
        print(f"  → Density: {density:.6f}")

        # Isolated nodes
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        num_isolated = torch.sum(degrees == 0).item()
        print(f"🧍 Isolated nodes: {num_isolated}")

        # Available attributes
        print(f"\n📦 Available attributes: {list(data.keys())}")
    diagnose_graph(data, parameters)

    os.makedirs(model_output_dir, exist_ok=True)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    number_total_nodes = data.num_nodes
    labels = [G.nodes[node_id]["text_label"] for node_id in node_list]  
    num_chromosome = labels.count("chromosome")
    num_plasmid = labels.count("plasmid")
    num_ambiguous = labels.count("ambiguous")
    num_unlabeled = labels.count("unlabeled")
    assert number_total_nodes == num_chromosome + num_plasmid + num_ambiguous + num_unlabeled

    print(labels[:50])


    print("Chromosome contigs:", num_chromosome,
          "Plasmid contigs:", num_plasmid,
          "Ambiguous contigs:", num_ambiguous,
          "Unlabeled contigs:", num_unlabeled)

    chromosome_weight = 1.0 
    plasmid_weight = float(parameters["plasmid_ambiguous_weight"])
    ambiguous_weight = plasmid_weight

    masks = []
    for node_id in node_list:
        label = G.nodes[node_id]["text_label"]
        if label == "unlabeled":
            masks.append(0.0)
        elif label == "chromosome":
            masks.append(chromosome_weight)
        elif label == "plasmid":
            masks.append(plasmid_weight)
        elif label == "ambiguous":
            masks.append(ambiguous_weight)
    masks = np.array(masks, dtype=float)

    print("Final masks list:", masks)

    seed_number = parameters["random_seed"] 
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number + 1)
    torch.manual_seed(seed_number + 2)
    
    masks_train = masks.copy()
    masks_validate = masks.copy()
    for i in range(len(masks)):
        if random.random() > 0.8:
            masks_train[i] = 0.0
        else:
            masks_validate[i] = 0.0

    print(masks_train[0:20])
    print(masks_validate[0:20])

    masks_train = torch.tensor(masks_train, dtype=torch.float32)
    masks_validate = torch.tensor(masks_validate, dtype=torch.float32)

    print(len(masks_train))
    print(len(masks_validate))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # Check for MPS
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}") # Optional: to confirm which device is being used
    data = data.to(device)

    print("🔍 Label breakdown (new format):")
    num_plasmid_actual = ((data.y[:, 0] == 1) & (data.y[:, 1] == 0)).sum().item()
    num_chromosome_actual = ((data.y[:, 0] == 0) & (data.y[:, 1] == 1)).sum().item()
    num_ambiguous_actual = ((data.y[:, 0] == 1) & (data.y[:, 1] == 1)).sum().item() 
    num_unlabeled_actual = ((data.y[:, 0] == 0) & (data.y[:, 1] == 0)).sum().item() 
    
    print(f"  Plasmid ([1,0]):      {num_plasmid_actual}")
    print(f"  Chromosome ([0,1]):   {num_chromosome_actual}")
    print(f"  Ambiguous ([1,1]):    {num_ambiguous_actual}")
    print(f"  Unlabeled ([0,0]):    {num_unlabeled_actual}")
    print(f"  Total label rows:     {len(data.y)}")

    masks_train = masks_train.to(device)
    masks_validate = masks_validate.to(device)
    
    study_name = parameters._params.get("optuna_study_name", "plASgraph_optimization")
    n_trials = parameters._params.get("optuna_n_trials", 20) 

    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: objective(trial, parameters, data, masks_train, masks_validate, device, model_output_dir, log_dir, node_list, G), 
                   n_trials=n_trials)

    print("\nOptuna study finished.")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Validation Loss): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")



    print("\n--- Generating Optuna Visualizations ---")
    optuna_viz_dir = os.path.join(model_output_dir, "optuna_visualizations")
    os.makedirs(optuna_viz_dir, exist_ok=True)

    # 1. Optimization History Plot
    try:
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_image(os.path.join(optuna_viz_dir, "optimization_history.png"))
        # To make it interactive (HTML):
        # fig_history.write_html(os.path.join(optuna_viz_dir, "optimization_history.html"))
        print(f"Saved Optuna optimization history plot to {optuna_viz_dir}")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    # 2. Parameter Importances Plot
    try:
        # This plot is most informative when there are enough completed trials
        if len(study.trials) > 1 : # Need at least 2 trials to compute importance
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_image(os.path.join(optuna_viz_dir, "param_importances.png"))
            # fig_importance.write_html(os.path.join(optuna_viz_dir, "param_importances.html"))
            print(f"Saved Optuna parameter importances plot to {optuna_viz_dir}")
        else:
            print("Skipping parameter importances plot: not enough trials completed.")
    except Exception as e:
        print(f"Could not generate parameter importances plot: {e}")

    # 3. Slice Plot (for each parameter)
    try:
        for param_name in best_trial.params.keys():
            fig_slice = vis.plot_slice(study, params=[param_name])
            fig_slice.write_image(os.path.join(optuna_viz_dir, f"slice_plot_{param_name}.png"))
            # fig_slice.write_html(os.path.join(optuna_viz_dir, f"slice_plot_{param_name}.html"))
        print(f"Saved Optuna slice plots to {optuna_viz_dir}")
    except Exception as e:
        print(f"Could not generate slice plots: {e}")

    # 4. Parallel Coordinate Plot
    try:
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.write_image(os.path.join(optuna_viz_dir, "parallel_coordinate.png"))
        # fig_parallel.write_html(os.path.join(optuna_viz_dir, "parallel_coordinate.html"))
        print(f"Saved Optuna parallel coordinate plot to {optuna_viz_dir}")
    except Exception as e:
        print(f"Could not generate parallel coordinate plot: {e}")

    # 5. Contour Plot (Example for two parameters if they exist)
    #    Choose parameters you expect to interact, e.g., learning_rate and l2_reg
    try:
        if 'learning_rate' in best_trial.params and 'l2_reg' in best_trial.params and len(study.trials) > 1:
            fig_contour = vis.plot_contour(study, params=['learning_rate', 'l2_reg'])
            fig_contour.write_image(os.path.join(optuna_viz_dir, "contour_plot_lr_l2reg.png"))
            # fig_contour.write_html(os.path.join(optuna_viz_dir, "contour_plot_lr_l2reg.html"))
            print(f"Saved Optuna contour plot for learning_rate and l2_reg to {optuna_viz_dir}")
        else:
            print("Skipping contour plot: 'learning_rate' or 'l2_reg' not in params, or not enough trials.")
    except Exception as e:
        print(f"Could not generate contour plot: {e}")

    # ... (rest of your script, e.g., final model retraining)



    best_trials_base_dir = os.path.join(model_output_dir, "trials")
    best_model_output_dir = os.path.join(best_trials_base_dir, f"trial_{best_trial.number}")
    # Then best_config_path will be correct:
    best_config_path = os.path.join(best_model_output_dir, config.DEFAULT_FILENAME)


    best_parameters = config.config(best_config_path) 

    # Dictionary-style access for best_parameters
    if best_parameters['model_type'] == 'GCNModel':
        model = GCNModel(best_parameters).to(device)
    elif best_parameters['model_type'] == 'GGNNModel':
        model = GGNNModel(best_parameters).to(device)
    else:
        raise ValueError(f"Unsupported model type in best trial: {best_parameters['model_type']}")
    
    optimizer = torch.optim.Adam( # Re-initialize optimizer for the new model instance
        model.parameters(),
        lr=best_parameters['learning_rate'],
        weight_decay=best_parameters['l2_reg']
    )

    if best_parameters["loss_function"] == "crossentropy":
        criterion = torch.nn.BCELoss(reduction='none') # Re-initialize criterion
    elif best_parameters["loss_function"] == "mse":
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise ValueError(f"Unsupported loss function: {best_parameters['loss_function']}")

    best_val_loss_retrain = float("inf")
    patience_retrain = 0
    final_retrained_model_path = os.path.join(model_output_dir, "final_retrained_best_model.pt")
    
    train_losses_final_retrain = []
    val_losses_final_retrain = []

    print(f"\n--- Starting Final Model Retraining ---")
    for epoch in range(parameters["epochs"]):
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
        fix_gradients(best_parameters, model)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(data)
            masked_outputs_val = val_outputs[valid_label_mask]
            masked_labels_val = data.y[valid_label_mask]
            masked_weights_val = masks_validate[valid_label_mask]

            loss_per_node_val = criterion(masked_outputs_val, masked_labels_val)
            val_loss = (loss_per_node_val.sum(dim=1) * masked_weights_val).sum() / masked_weights_val.sum()
        
        train_losses_final_retrain.append(train_loss.item())
        val_losses_final_retrain.append(val_loss.item())

        print(f"Retraining Epoch {epoch+1}/{parameters['epochs']} - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss_retrain:
            best_val_loss_retrain = val_loss.item()
            patience_retrain = 0
            torch.save(model.state_dict(), final_retrained_model_path)
        else:
            patience_retrain += 1
            early_stopping_patience_final = best_parameters._params.get("early_stopping_patience_retrain", best_parameters["early_stopping_patience"])
            if patience_retrain >= early_stopping_patience_final:
                print(f"Early stopping triggered during final retraining at epoch {epoch+1}.")
                break
    
    if train_losses_final_retrain and val_losses_final_retrain:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_final_retrain, label='Retrain Train Loss')
        plt.plot(val_losses_final_retrain, label='Retrain Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Best Model Retraining (Trial {best_trial.number}) - Loss')
        plt.legend()
        plt.grid(True)
        y_max_val = max(1, max(train_losses_final_retrain) if train_losses_final_retrain else 1, max(val_losses_final_retrain) if val_losses_final_retrain else 1) * 1.1
        plt.ylim(0, y_max_val if y_max_val > 0 else 1)
        retrain_loss_plot_path = os.path.join(model_output_dir, "retraining_loss_over_epochs.png")
        plt.savefig(retrain_loss_plot_path)
        plt.close()

    print(f"\nFinal retraining finished. Loading best model state from {final_retrained_model_path} for evaluation.")
    if os.path.exists(final_retrained_model_path):
        model.load_state_dict(torch.load(final_retrained_model_path, map_location=device))
    else:
        print(f"WARNING: Best retrained model path not found: {final_retrained_model_path}. Using model from last epoch of retraining.")
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
        
        best_parameters.write_yaml(os.path.join(model_output_dir, config_with_thresholds_filename))
        print(f"Optimal thresholds saved to config file in {model_output_dir} as {config_with_thresholds_filename}")


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
