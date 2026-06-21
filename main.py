"""Extended experiment entry point built on the public CA-TCC project.

It preserves the authors' encoder, temporal/contextual objectives and CA stages,
while parameterizing 1%, 5%, 10% experiments and adding ablations.
"""

from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import time


def _early_cpu_thread_count(default=4):
    try:
        index = sys.argv.index('--cpu_threads')
        return max(1, int(sys.argv[index + 1]))
    except (ValueError, IndexError):
        return default


_EARLY_CPU_THREADS = _early_cpu_thread_count()
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = str(_EARLY_CPU_THREADS)

import torch

from dataloader.dataloader import data_generator, safe_torch_load
from models.model import base_Model
from models.TC import TC
from trainer.trainer import (
    evaluate_classifier,
    fit_contrastive,
    fit_supervised,
    generate_pseudo_labels,
)
from utils import (
    fix_randomness,
    get_logger,
    load_checkpoint,
    save_classification_metrics,
    set_encoder_trainable,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_description', default='SpokenArabicDigits_experiment')
    parser.add_argument('--run_description', default='baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--training_mode', default='self_supervised')
    parser.add_argument('--selected_dataset', default='SpokenArabicDigits')
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--logs_save_dir', default='experiments_logs')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu_threads', type=int, default=4)
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--pseudo_output', default=None, help='Filename for generated pseudo-label dataset')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--label_percentage', type=int, choices=[1, 5, 10, 50, 75], default=None)
    parser.add_argument('--confidence_threshold', type=float, default=0.0)
    parser.add_argument('--method_name', default=None)
    parser.add_argument('--result_role', choices=['final','intermediate'], default='final')
    return parser.parse_args()


def infer_ratio(mode, explicit):
    if explicit is not None:
        return explicit
    import re
    match = re.search(r'_(1|5|10|50|75)p(?:$|_)', mode)
    return int(match.group(1)) if match else None


def device_from_arg(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    requested = torch.device(value)
    if requested.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA was requested but is unavailable; falling back to CPU.')
        return torch.device('cpu')
    return requested


def main():
    args = parse_args()
    fix_randomness(args.seed)
    device = device_from_arg(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    config_module = importlib.import_module(f'config_files.{args.selected_dataset}_Configs')
    config = config_module.Config()
    if args.epochs is not None:
        config.num_epoch = args.epochs

    ratio = infer_ratio(args.training_mode, args.label_percentage)
    output_dir = Path(args.logs_save_dir) / args.experiment_description / args.run_description / f'{args.training_mode}_seed_{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir / 'run.log')
    logger.info(f'Dataset={args.selected_dataset} mode={args.training_mode} seed={args.seed} device={device}')

    data_dir = Path(args.data_path) / args.selected_dataset
    model = base_Model(config).to(device)
    temporal_model = TC(config, device).to(device)

    if args.load_checkpoint:
        exclude_classifier = (
            args.training_mode.startswith('ft_')
            or args.training_mode.startswith('train_linear')
            or args.training_mode.startswith('SupCon')
        ) and not args.training_mode.startswith('gen_pseudo')
        info = load_checkpoint(
            args.load_checkpoint,
            model,
            temporal_model,
            exclude_classifier=exclude_classifier,
            strict=False,
        )
        logger.info(f'Loaded checkpoint {args.load_checkpoint}')
        logger.info(f"Missing keys={info['missing_keys']} unexpected={info['unexpected_keys']}")

    if args.training_mode.startswith('gen_pseudo_labels'):
        if ratio is None:
            raise ValueError('gen_pseudo_labels requires --label_percentage or suffix such as _5p')
        if not args.load_checkpoint:
            raise ValueError('Pseudo-label generation requires --load_checkpoint from a fine-tuned model.')
        # Reload full classifier including logits because exclude_classifier is false in this branch.
        load_checkpoint(args.load_checkpoint, model, temporal_model, exclude_classifier=False, strict=False)
        full_payload = safe_torch_load(data_dir / 'train.pt')
        labeled_payload = safe_torch_load(data_dir / f'train_{ratio}perc.pt')
        audit = generate_pseudo_labels(
            model, full_payload, labeled_payload, device,
            data_dir / (args.pseudo_output or f'pseudo_train_{ratio}perc.pt'),
            confidence_threshold=args.confidence_threshold,
        )
        (output_dir / 'pseudo_label_metrics.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        logger.info(json.dumps(audit, indent=2))
        return

    train_loader, val_loader, test_loader = data_generator(
        data_dir, config, args.training_mode, train_file=args.train_file
    )
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
    )
    temporal_optimizer = torch.optim.Adam(
        temporal_model.parameters(), lr=config.lr,
        betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
    )

    if args.training_mode == 'self_supervised' or args.training_mode.startswith('SupCon'):
        fit_contrastive(
            model, temporal_model, train_loader, model_optimizer, temporal_optimizer,
            device, config, output_dir, logger, args.training_mode, epochs=args.epochs,
        )
        return

    if args.training_mode.startswith('train_linear'):
        set_encoder_trainable(model, False)
        model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
        )
    else:
        set_encoder_trainable(model, True)
        model_optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr,
            betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
        )

    history = fit_supervised(
        model, temporal_model, train_loader, val_loader, model_optimizer,
        device, config, output_dir, logger, epochs=args.epochs,
    )
    best = output_dir / 'saved_models/ckp_best.pt'
    if best.exists():
        load_checkpoint(best, model, temporal_model, exclude_classifier=False, strict=False)
    test = evaluate_classifier(model, test_loader, device)
    metadata = {
        'experiment': args.experiment_description,
        'run_description': args.run_description,
        'training_mode': args.training_mode,
        'method': args.method_name or args.run_description,
        'label_percentage': ratio,
        'seed': args.seed,
        'device': str(device),
        'result_role': args.result_role,
        'epochs_completed': int(len(history)),
        'current_stage_training_seconds': float(sum(row.get('seconds', 0.0) for row in history)),
        'best_val_macro_f1': float(max((row.get('val_macro_f1', 0.0) for row in history), default=0.0)),
        'n_parameters_total': int(sum(parameter.numel() for parameter in model.parameters())),
        'n_parameters_trainable': int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        'loaded_checkpoint': args.load_checkpoint,
    }
    metrics = save_classification_metrics(
        test['targets'], test['predictions'], test['probabilities'], output_dir, metadata
    )
    logger.info(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
