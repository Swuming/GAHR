import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report
)


class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        def round_value(v):
            if isinstance(v, (float, np.float32, np.float64)):
                return round(float(v), 4)
            elif isinstance(v, (list, tuple)):
                return [round_value(i) for i in v]
            elif isinstance(v, dict):
                return {k: round_value(val) for k, val in v.items()}
            return v
        if isinstance(obj, dict):
            return super().encode({k: round_value(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return super().encode([round_value(i) for i in obj])
        return super().encode(obj)


class MetricsCalculator:
    @staticmethod
    def calculate_metrics(preds, labels, threshold=0.5):
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        pred_binary = (preds_np > threshold).astype(int)
        labels_int = labels_np.astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels_int, pred_binary, labels=[0, 1], zero_division=0)
        real_precision, fake_precision = precision
        real_recall, fake_recall = recall
        real_f1, fake_f1 = f1
        metrics = {
            'accuracy': accuracy_score(labels_int, pred_binary),
            'fake_precision': fake_precision,
            'fake_recall': fake_recall,
            'fake_f1': fake_f1,
            'real_precision': real_precision,
            'real_recall': real_recall,
            'real_f1': real_f1,
            'auc': roc_auc_score(labels_int, preds_np) if len(np.unique(labels_int)) > 1 else 0.0,
            'precision': fake_precision,
            'recall': fake_recall,
            'f1': fake_f1,
        }
        return metrics

    @staticmethod
    def get_classification_report(preds, labels, threshold=0.5):
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        pred_binary = (preds_np > threshold).astype(int)
        labels_int = labels_np.astype(int)
        return classification_report(
            labels_int,
            pred_binary,
            labels=[0, 1],
            target_names=['Real News', 'Fake News'],
            digits=4,
            zero_division=0
        )


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class ModelCheckpoint:
    def __init__(self, path, monitor='accuracy', mode='max'):
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def save_if_best(self, model, epoch, metrics):
        score = metrics.get(self.monitor, 0)
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == 'max' and score > self.best_score:
            is_best = True
        elif self.mode == 'min' and score < self.best_score:
            is_best = True
        if is_best:
            self.best_score = score
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            model_state_dict = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                },
                self.path
            )
        return is_best

    def load_best_model(self, model):
        if not os.path.exists(self.path):
            print(f"警告: 未找到模型权重文件: {self.path}")
            return 0, {}
        checkpoint = torch.load(self.path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


class Logger:
    def __init__(self, log_dir, mode='train'):
        self.log_dir = log_dir
        self.mode = mode
        os.makedirs(log_dir, exist_ok=True)
        self.train_log = []
        self.val_log = []
        self.best_acc = 0.0
        self.detailed_file = os.path.join(log_dir, 'detailed_training_report.txt')
        self.best_file = os.path.join(log_dir, 'best_accuracy_records.txt')

        if mode == 'train':
            open(self.detailed_file, 'w', encoding='utf-8').close()
            open(self.best_file, 'w', encoding='utf-8').close()

    def log_epoch(self, epoch, train_metrics, val_metrics, lr, report=None):
        self.train_log.append({
            'epoch': epoch,
            'lr': lr,
            **train_metrics
        })
        self.val_log.append({
            'epoch': epoch,
            **val_metrics
        })
        with open(os.path.join(self.log_dir, 'train_log.json'), 'w', encoding='utf-8') as f:
            json.dump(
                self.train_log,
                f,
                indent=2,
                ensure_ascii=False,
                cls=CustomJSONEncoder
            )
        with open(os.path.join(self.log_dir, 'val_log.json'), 'w', encoding='utf-8') as f:
            json.dump(
                self.val_log,
                f,
                indent=2,
                ensure_ascii=False,
                cls=CustomJSONEncoder
            )
        self._save_detailed(epoch, train_metrics, val_metrics, report)
        current_acc = val_metrics.get('accuracy', 0.0)
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self._save_best(epoch, current_acc)

    def _save_detailed(self, epoch, train_metrics, val_metrics, report=None):
        with open(self.detailed_file, 'a', encoding='utf-8') as f:
            f.write(f'Epoch {epoch}:\n')
            f.write(
                f'  Train - '
                f'Acc: {train_metrics.get("accuracy", 0):.4f}, '
                f'F1: {train_metrics.get("f1", 0):.4f}, '
                f'AUC: {train_metrics.get("auc", 0):.4f}, '
                f'Loss: {train_metrics.get("veracity_loss", 0):.4f}\n'
            )
            f.write(
                f'  Val   - '
                f'Acc: {val_metrics.get("accuracy", 0):.4f}, '
                f'F1: {val_metrics.get("f1", 0):.4f}, '
                f'AUC: {val_metrics.get("auc", 0):.4f}, '
                f'Loss: {val_metrics.get("veracity_loss", 0):.4f}\n'
            )
            if report:
                f.write(report + '\n')
            f.write('-' * 60 + '\n')

    def _save_best(self, epoch, acc):
        with open(self.best_file, 'a', encoding='utf-8') as f:
            f.write(
                f'Best Record Updated - '
                f'Epoch: {epoch}, '
                f'Best Acc: {acc:.4f}, '
                f'Time: {datetime.now().isoformat()}\n'
            )


def create_optimizer_scheduler(model, config, total_steps):
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler


def save_predictions(preds, labels, texts, image_paths, output_path,):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    results = []
    for i in range(len(preds_np)):
        pred_score = float(preds_np[i])
        label = int(labels_np[i])
        pred_label = int(pred_score > 0.5)
        item = {
            'prediction': pred_score,
            'pred_label': pred_label,
            'label': label,
            'correct': int(pred_label == label),
            'text': texts[i] if i < len(texts) else '',
            'image_path': image_paths[i] if i < len(image_paths) else '',
        }
        results.append(item)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            cls=CustomJSONEncoder
        )


def setup_logging(config):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = getattr(config, 'output_root', 'outputs')
    log_dir = os.path.join(
        output_root,
        'logs',
        timestamp,
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
