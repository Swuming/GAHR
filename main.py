import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from dataset import create_feature_loaders
from model import CombinedLoss, MultiModalConsistencyModel
from utils import (
    CustomJSONEncoder,
    EarlyStopping,
    Logger,
    MetricsCalculator,
    ModelCheckpoint,
    create_optimizer_scheduler,
    save_predictions,
    setup_logging,
)


LOSS_NAMES = ('total_loss', 'veracity_loss', 'generativity_loss')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_inputs(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _forward(model, inputs):
    return model(
        text_hidden=inputs['text_hidden'],
        vision_hidden=inputs['vision_hidden'],
        clip_text_global=inputs['clip_text_global'],
        clip_image_global=inputs['clip_image_global'],
        clip_text_local=inputs['clip_text_local'],
        clip_vision_local=inputs['clip_vision_local'],
        attention_mask=inputs['attention_mask'],
        clip_attention_mask=inputs['clip_attention_mask'],
    )


def _empty_metrics():
    metrics = {
        'accuracy': 0.0,
        'fake_precision': 0.0,
        'fake_recall': 0.0,
        'fake_f1': 0.0,
        'real_precision': 0.0,
        'real_recall': 0.0,
        'real_f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'auc': 0.0,
    }
    metrics.update({name: 0.0 for name in LOSS_NAMES})
    return metrics


def _mean_loss_metrics(loss_log):
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in loss_log.items()
    }


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, config, scaler):
    model.train()
    predictions, labels = [], []
    loss_log = {name: [] for name in LOSS_NAMES}
    optimizer.zero_grad()
    for batch_index, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch} Training')):
        inputs = _model_inputs(batch, device)
        batch_labels = inputs['label']
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = _forward(model, inputs)
            loss_dict = criterion(outputs, batch_labels, inputs)
            loss = loss_dict['total_loss'] / config.accumulation_steps
        scaler.scale(loss).backward()
        should_step = (
            (batch_index + 1) % config.accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if previous_scale <= scaler.get_scale():
                scheduler.step()
        predictions.extend(outputs['prediction'].detach().float().cpu().tolist())
        labels.extend(batch_labels.detach().float().cpu().tolist())
        for name in LOSS_NAMES:
            loss_log[name].append(loss_dict[name].item())
    if not predictions:
        return _empty_metrics()
    metrics = MetricsCalculator.calculate_metrics(torch.tensor(predictions), torch.tensor(labels))
    metrics.update(_mean_loss_metrics(loss_log))
    return metrics


def evaluate(model, loader, criterion, device, epoch, description='Validation'):
    model.eval()
    predictions, labels, texts, image_paths = [], [], [], []
    loss_log = {name: [] for name in LOSS_NAMES}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Epoch {epoch} {description}'):
            inputs = _model_inputs(batch, device)
            batch_labels = inputs['label']
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = _forward(model, inputs)
                loss_dict = criterion(outputs, batch_labels, inputs)
            predictions.extend(outputs['prediction'].detach().float().cpu().tolist())
            labels.extend(batch_labels.detach().float().cpu().tolist())
            texts.extend(batch['text'])
            image_paths.extend(batch['image_path'])
            for name in LOSS_NAMES:
                loss_log[name].append(loss_dict[name].item())
    if not predictions:
        return _empty_metrics(), [], [], [], [], ''
    metrics = MetricsCalculator.calculate_metrics(torch.tensor(predictions), torch.tensor(labels))
    metrics.update(_mean_loss_metrics(loss_log))
    report = MetricsCalculator.get_classification_report(torch.tensor(predictions), torch.tensor(labels))
    return metrics, predictions, labels, texts, image_paths, report


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate GAHR.')
    parser.add_argument('--mode', choices=['train', 'test', 'train_test'], default='train_test')
    parser.add_argument('--feature_dir', default='features')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--language', choices=['zh', 'en'], default=None)
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--accumulation_steps', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--monitor', choices=['accuracy', 'f1'], default='accuracy')
    return parser.parse_args()


def build_config(args):
    config = Config(language=args.language)
    overrides = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'learning_rate': args.learning_rate,
        'accumulation_steps': args.accumulation_steps,
    }
    for name, value in overrides.items():
        if value is not None:
            if value <= 0:
                raise ValueError(f'{name} must be positive')
            setattr(config, name, value)
    if args.output_dir is not None:
        config.output_root = args.output_dir
        config.best_model_path = os.path.join(args.output_dir, 'model', 'best_model.pth')
    return config


def main():
    args = parse_args()
    set_random_seed(args.seed)
    config = build_config(args)
    requested_splits = ('test',) if args.mode == 'test' else ('train', 'val', 'test')
    train_loader, val_loader, test_loader = create_feature_loaders(args.feature_dir, config, splits=requested_splits)
    if args.mode in ('train', 'train_test') and (train_loader is None or val_loader is None):
        raise FileNotFoundError('Training requires train_features.pt and val_features.pt.')
    if args.mode == 'test' and test_loader is None:
        raise FileNotFoundError('Testing requires test_features.pt.')
    model = MultiModalConsistencyModel(config).to(config.device)
    criterion = CombinedLoss(config)
    checkpoint = ModelCheckpoint(config.best_model_path, monitor=args.monitor, mode='max')

    if args.mode in ('train', 'train_test'):
        log_dir = setup_logging(config)
        steps_per_epoch = (len(train_loader) + config.accumulation_steps - 1) // config.accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs
        optimizer, scheduler = create_optimizer_scheduler(model, config, total_steps)
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        early_stopping = EarlyStopping(config.patience, config.min_delta, mode='max')
        logger = Logger(log_dir, 'train')
        if args.resume:
            checkpoint.load_best_model(model)
        print('\nTraining GAHR...')
        best_metric = 0.0
        for epoch in range(1, config.num_epochs + 1):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.device, epoch, config, scaler,)
            val_metrics, predictions, labels, texts, image_paths, report = evaluate(model, val_loader, criterion, config.device, epoch)
            print(
                f'Epoch {epoch}: train Acc={train_metrics["accuracy"]:.4f}, '
                f'val Acc={val_metrics["accuracy"]:.4f}, val F1={val_metrics["f1"]:.4f}'
            )
            logger.log_epoch(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'], report)
            if checkpoint.save_if_best(model, epoch, val_metrics):
                best_metric = val_metrics[args.monitor]
                save_predictions(
                    torch.tensor(predictions), torch.tensor(labels), texts, image_paths,
                    os.path.join(log_dir, 'best_predictions.json'),
                )
            if early_stopping(val_metrics[args.monitor]):
                print(f'Early stopping at epoch {epoch}.')
                break
        print(f'Finished training. Best validation {args.monitor}: {best_metric:.4f}')

    if args.mode in ('test', 'train_test'):
        if args.mode == 'test':
            log_dir = setup_logging(config)
        if test_loader is None:
            _, _, test_loader = create_feature_loaders(args.feature_dir, config, splits=('test',))
        if test_loader is None:
            raise FileNotFoundError('Testing requires test_features.pt.')
        if not os.path.exists(config.best_model_path):
            raise FileNotFoundError(f'Best checkpoint not found: {config.best_model_path}')
        checkpoint.load_best_model(model)
        test_metrics, predictions, labels, texts, image_paths, report = evaluate(model, test_loader, criterion, config.device, 0, 'Test')
        print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
        print(f'Test Fake F1: {test_metrics["fake_f1"]:.4f}')
        print(f'Test AUC: {test_metrics["auc"]:.4f}')
        print(report)
        save_predictions(
            torch.tensor(predictions), torch.tensor(labels), texts, image_paths,
            os.path.join(log_dir, 'test_predictions.json'),
        )
        with open(os.path.join(log_dir, 'test_metrics.json'), 'w', encoding='utf-8') as output_file:
            json.dump(test_metrics, output_file, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)


if __name__ == '__main__':
    main()
