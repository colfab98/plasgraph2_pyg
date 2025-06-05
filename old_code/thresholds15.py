"""Module for changing setting optimal threshold wrt F1 measure on validation set 
and applying them to new data"""

import numpy as np

import os
# import architecture

# ... (imports remain the same) ...
import torch # Add torch import
import torch.nn.functional as F # Add F if you need to manually apply sigmoid/softmax if model output is logits

# Note: The provided `architecture.py` now applies `self.output_activation` in `forward`.
# So, `model(graph)` will return probabilities directly.

def set_thresholds(model, data, masks_validate, parameters, log_dir=None):
    """Set optimal thresholds (plasmid, chromosome) by maximizing F1 on validation set,
    and store them in the parameters object. This version works with PyTorch models/data."""

    # Ensure model is in evaluation mode
    model.eval()
    
    # Move data to model's device (if not already there)
    device = next(model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        # Get raw predictions/probabilities from the model for the entire graph
        # model.forward() now applies the output activation, so `outputs` are already probabilities
        outputs = model(data) 
        
        # Extract true labels for validation nodes
        # Use data.y for true labels and masks_validate for selection
        
        # --- Process Plasmid ---
        # Select only validation set nodes for plasmid output
        labels_plasmid_val = data.y[masks_validate.bool(), 0]
        probs_plasmid_val = outputs[masks_validate.bool(), 0]

        # In your train9.py, `valid_eval_mask_plasmid = labels_plasmid_val_masked != -1` was used.
        # This typically means labels could be -1 (e.g., ignored or missing).
        # Assuming `data.y` is clean (0/1/other numerical labels), and `masks_validate` already
        # handles the train/val split (0 for train, >0 for val), then you just need to
        # filter out labels that are specifically meant to be ignored from F1 calculation.
        # If your labels are always 0 or 1 for actual classes, and 0 for unlabeled, and 1,1 for ambiguous,
        # then you primarily care about samples where the label is 0 or 1 for the *specific class*.
        # The `masks_validate` array should handle the actual samples to consider.
        
        # Filter for actual 0/1 true labels where relevant (excluding [0,0] from being target 0/1)
        # This part depends on how you want to treat the 0,0 labels (unlabeled) in F1 score.
        # If the `masks_validate` already sets their weight to 0, then `score_thresholds` should handle it.
        # The `score_thresholds` in `thresholds.py` explicitly uses `if weights[i] > 0:`
        # So, we just need to pass the y_true and y_pred for all relevant nodes from the validation set
        # with their corresponding masks_validate.
        
        # Convert to numpy arrays for sklearn's f1_score
        y_true_plasmid = labels_plasmid_val.cpu().numpy()
        y_probs_plasmid = probs_plasmid_val.cpu().numpy()
        sample_weight_plasmid = masks_validate[masks_validate.bool()].cpu().numpy() # weights for validation nodes

        # Call score_thresholds which is designed to work with y_true (0/1), y_pred (float), and weights
        # Note: score_thresholds in the original thresholds.py computes F1 iteratively.
        # It's not sklearn.f1_score. We need to check if it already uses sample_weight.
        # Checking thresholds.py: score_thresholds takes y_true, y_pred, weights and filters `if weights[i] > 0`
        # and then counts tp based on `if pairs[i][0] > 0.5`. This is compatible.
        
        plasmid_scores = score_thresholds(y_true_plasmid, y_probs_plasmid, sample_weight_plasmid)
        store_best(plasmid_scores, parameters, 'plasmid_threshold', log_dir)

        # --- Process Chromosome ---
        # Select only validation set nodes for chromosome output
        labels_chromosome_val = data.y[masks_validate.bool(), 1]
        probs_chromosome_val = outputs[masks_validate.bool(), 1]

        y_true_chromosome = labels_chromosome_val.cpu().numpy()
        y_probs_chromosome = probs_chromosome_val.cpu().numpy()
        sample_weight_chromosome = masks_validate[masks_validate.bool()].cpu().numpy()

        chromosome_scores = score_thresholds(y_true_chromosome, y_probs_chromosome, sample_weight_chromosome)
        store_best(chromosome_scores, parameters, 'chromosome_threshold', log_dir)



def apply_thresholds(y, parameters):
    """Apply thresholds during testing, return transformed scores so that 0.5 corresponds to threshold"""
    columns = []
    for (column_idx, which_parameter) in [(0, 'plasmid_threshold'), (1, 'chromosome_threshold')]:
        threshold = parameters[which_parameter]
        orig_column = y[:, column_idx]
        # apply the scaling function with different parameters for small and large numbers
        new_column = np.piecewise(
            orig_column,
            [orig_column < threshold, orig_column >= threshold],
            [lambda x : scale_number(x, 0, threshold, 0, 0.5), lambda x : scale_number(x, threshold, 1, 0.5, 1)]
        )
        columns.append(new_column)

    y_new = np.array(columns).transpose()
    return y_new

def scale_number(x, s1, e1, s2, e2):
    """Scale number x so that interval (s1,e1) is transformed to (s2, e2)"""

    factor = (e2 - s2) / (e1 - s1)
    return (x - s1) * factor + s2

def store_best(scores, parameters, which, log_dir):
    """store the optimal threshold for one output in parameter and if requested, print all thresholds to a log file"""
    # scores is a list of pairs threshold, F1 score
    if len(scores) > 0:
        # find index of maximum in scores[*][1]
        maxindex = max(range(len(scores)), key = lambda i : scores[i][1])
        # corrsponding item in scores[*][0] is the threshold
        threshold = scores[maxindex][0]
    else:
        # is input array empty, use default 0.5
        threshold = 0.5
    # store the found threshold
    parameters[which] = float(threshold)

    if log_dir is not None:
        # store thresholds and F1 scores
        filename = os.path.join(log_dir, which + ".csv")
        with open(filename, 'wt') as file:
            print(f"{which},f1", file=file)
            for x in scores:
                print(",".join(str(value) for value in x), file=file)

def score_thresholds(y_true, y_pred, weights):
    """Compute F1 score of all thresholds for one output (plasmid or chromosome)"""
    # compute vector weight and check that all are the same
    length = y_true.shape[0]
    assert tuple(y_true.shape) == (length,)
    assert tuple(y_pred.shape) == (length,)
    assert tuple(weights.shape) == (length,)
    # get data points with non-zero weight
    pairs = []
    for i in range(length):
        if weights[i] > 0:
            pairs.append((y_true[i], y_pred[i]))
    pairs.sort(key=lambda x : x[1], reverse=True)
    
    # count all positives in true labels
    pos = 0
    for pair in pairs:
        if pair[0] > 0.5:
            pos += 1

    scores = []
    tp = 0
    for i in range(len(pairs)):
        # increase true positives if true label is 
        if pairs[i][0] > 0.5:
            tp += 1
        if i > 0 and pairs[i][1] < pairs[i-1][1]:
            recall = tp / pos
            precision = tp / (i+1)
            if (precision + recall) == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            threshold = (pairs[i-1][1] + pairs[i][1]) / 2
            scores.append((threshold, f1))
    
    return scores
