"""Experiment utilities and reproducible metric export."""

from pathlib import Path
import json
import logging
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def fix_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path.resolve()))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_encoder_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable if not name.startswith('logits.') else True


def load_checkpoint(path, model, temporal_model=None, exclude_classifier=False, strict=False):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    if exclude_classifier:
        state = {key: value for key, value in state.items() if not key.startswith('logits.')}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if temporal_model is not None and 'temporal_contr_model_state_dict' in checkpoint:
        temporal_model.load_state_dict(checkpoint['temporal_contr_model_state_dict'], strict=False)
    return {'missing_keys': list(missing), 'unexpected_keys': list(unexpected), 'raw': checkpoint}


def save_checkpoint(path, model, temporal_model, epoch, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'temporal_contr_model_state_dict': temporal_model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_classification_metrics(y_true, y_pred, probabilities, output_dir, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics = {
        **metadata,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'n_test_samples': int(len(y_true)),
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame([metrics]).to_csv(output_dir / 'metrics.csv', index=False)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / 'classification_report.csv')
    matrix = confusion_matrix(y_true, y_pred)
    np.save(output_dir / 'confusion_matrix.npy', matrix)
    np.save(output_dir / 'true_labels.npy', y_true)
    np.save(output_dir / 'predicted_labels.npy', y_pred)
    if probabilities is not None:
        np.save(output_dir / 'probabilities.npy', np.asarray(probabilities))
    return metrics
