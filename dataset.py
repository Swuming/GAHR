import os

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


GENERATIVE_CATEGORIES = {
    'NRN': (0.0, 0.0),
    'NFN': (0.0, 0.0),
    'ICRN': (0.0, 1.0),
    'ICFN': (0.0, 1.0),
    'TCRN': (1.0, 0.0),
    'TCFN': (1.0, 0.0),
}


def _normalize_category(category):
    if category is None:
        return None
    text = str(category).strip().upper().replace('_', ' ').replace('-', ' ')
    compact = text.replace(' ', '')
    for name in GENERATIVE_CATEGORIES:
        if name in compact:
            return name
    aliases = {
        'NATIVE REAL NEWS': 'NRN',
        'NATIVE FAKE NEWS': 'NFN',
        'IMAGE CONDITIONED REAL NEWS': 'ICRN',
        'IMAGE CONDITIONED FAKE NEWS': 'ICFN',
        'TEXT CONDITIONED REAL NEWS': 'TCRN',
        'TEXT CONDITIONED FAKE NEWS': 'TCFN',
    }
    return aliases.get(text)


def derive_generativity_targets(category):
    normalized = _normalize_category(category)
    if normalized is None:
        return 0.0, 0.0, 0.0
    image_target, text_target = GENERATIVE_CATEGORIES[normalized]
    return image_target, text_target, 1.0


class FeatureDataset(Dataset):
    def __init__(self, feature_path, split='train'):
        self.items = torch.load(feature_path, map_location='cpu', weights_only=False)
        print(f'[{split.upper()}] Loaded {len(self.items)} feature records.')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image_target, text_target, supervision_mask = derive_generativity_targets(item.get('category'))
        return {
            'text_hidden': item['text_hidden'],
            'vision_hidden': item['vision_hidden'],
            'clip_text_global': item['clip_text_global'],
            'clip_image_global': item['clip_image_global'],
            'clip_text_local': item['clip_text_local'],
            'clip_vision_local': item['clip_vision_local'],
            'attention_mask': item['attention_mask'].long(),
            'clip_attention_mask': item['clip_attention_mask'].long(),
            'label': torch.tensor(item['label'], dtype=torch.float32),
            'image_generativity_label': torch.tensor(image_target, dtype=torch.float32),
            'text_generativity_label': torch.tensor(text_target, dtype=torch.float32),
            'generativity_mask': torch.tensor(supervision_mask, dtype=torch.float32),
            'text': item.get('text', ''),
            'image_path': item.get('image_path', ''),
        }


def feature_collate_fn(batch):
    tensor_keys = (
        'text_hidden', 'vision_hidden', 'clip_text_global', 'clip_image_global',
        'clip_text_local', 'clip_vision_local', 'attention_mask',
        'clip_attention_mask', 'label', 'image_generativity_label',
        'text_generativity_label', 'generativity_mask',
    )
    collated = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    collated['text'] = [item['text'] for item in batch]
    collated['image_path'] = [item['image_path'] for item in batch]
    return collated


def create_feature_loaders(feature_dir, config: Config, splits=None):
    requested_splits = set(splits or ('train', 'val', 'test'))
    loaders = []
    for split in ('train', 'val', 'test'):
        feature_path = os.path.join(feature_dir, f'{split}_features.pt')
        if split not in requested_splits or not os.path.exists(feature_path):
            loaders.append(None)
            continue
        dataset = FeatureDataset(feature_path, split=split)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=(split == 'train'),
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                collate_fn=feature_collate_fn,
            )
        )
    return tuple(loaders)
