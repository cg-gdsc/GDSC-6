"""
This is the evaluation module.

This module contains functions that help with evaluation of the Huggingface model.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Union, List, Any


def make_predictions(examples: torch.Tensor, 
                     model: torch.nn.Module, 
                     device: Union[str, torch.device],
                     labels: torch.Tensor = None) -> Dict[str, Union[List[str], np.ndarray]]:
    """
    Generates predictions and loss values for a given batch of examples and labels with the use of a provided model.

    Args:
        examples (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing input examples.
        labels (torch.Tensor): A tensor of shape (batch_size,) containing ground-truth labels.
        model (torch.nn.Module): A PyTorch model to use for making predictions.
        device (str or torch.device): The device to use for running the model (e.g. 'cpu' or 'cuda').

    Returns:
        dict: A dictionary containing one or two keys: 'predicted_class_id' (always) and 'loss' (optional).
        'predicted_class_id' is a list of strings representing the predicted class IDs for each example.
        'loss' is a numpy array of shape (batch_size, num_labels) containing the loss values for each example.
    """
    model = model.to(device)
    with torch.no_grad():
        logits = model(examples.to(device)).logits
    predicted_class_id = [str(torch.argmax(item).item()) for item in logits]
    if isinstance(labels, torch.Tensor):
        loss = torch.nn.functional.cross_entropy(logits.view(-1, model.num_labels), labels.to(device).view(-1), reduction="none")
        loss = loss.view(len(examples), -1).cpu().numpy()

        return {'predicted_class_id': predicted_class_id, 'loss': loss}
    else:
        return {'predicted_class_id': predicted_class_id}


def compute_metrics(pred: Any) -> Dict[str, float]:
    """
    Computes the accuracy, F1 score, precision, and recall for a set of predictions.

    Args:
        pred (Any): A set of predictions, as returned by a Hugging Face Trainer.

    Returns:
        Dict[str, float]: A dictionary containing four keys: 'accuracy', 'f1', 'precision', and 'recall',
        each with a float value representing the corresponding metric.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def plot_confusion_matrix(y_true: np.ndarray, 
                          y_preds: np.ndarray, 
                          normalize: str = "true") -> None:
    """
    Plot a confusion matrix.

    Parameters:
        y_true (np.ndarray): The true labels.
        y_preds (np.ndarray): The predicted labels.
        normalize (str, optional): Determines if the confusion matrix should be normalized. 
            Valid values are "true" (default) to normalize over the true labels, 
            "pred" to normalize over the predicted labels, "all" to normalize over the whole population
            or None to not normalize the matrix.

    Returns:
        None: This function doesn't return anything. It plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_preds, normalize=normalize)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, annot=True)
    plt.xlabel("predicted labels")
    plt.ylabel("true labels")
    if normalize=='true':
        plt.title("Confusion matrix normalized over true labels")
    elif normalize=='pred':
        plt.title("Confusion matrix normalized over predicted labels")
    elif normalize=='all':
        plt.title("Confusion matrix normalized over whole population")
    else:
        plt.title("Confusion matrix")
    