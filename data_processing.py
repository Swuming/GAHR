import argparse
import os

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    ChineseCLIPModel,
    CLIPModel,
    AutoTokenizer,
    AutoImageProcessor,
    ChineseCLIPProcessor,
    CLIPProcessor
)
from config import Config


class RawImageTextDataset(Dataset):
    def __init__(self, csv_path, image_root, config: Config, split='train'):
        self.config = config
        self.image_root = image_root
        print(f'[{split.upper()}] Loading raw records from {csv_path}')
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        self.vision_processor = AutoImageProcessor.from_pretrained(config.vision_model_name)
        if 'chinese' in config.clip_model_name.lower():
            self.clip_processor = ChineseCLIPProcessor.from_pretrained(config.clip_model_name)
        else:
            self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)

    def __len__(self):
        return len(self.df)

    def _get_default_image(self):
        return Image.new('RGB', (self.config.image_size, self.config.image_size), (0, 0, 0))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = float(row['label'])
        file_name = row.get('image', row.get('image_path', None))
        if not isinstance(file_name, str):
            file_name = "missing.jpg"
        image_path = os.path.join(self.image_root, file_name)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            image = self._get_default_image()
        category = row.get('category', row.get('type', 'unknown'))
        if pd.isna(category):
            category = 'unknown'
        category = str(category)
        text_inputs = self.tokenizer(
            text,
            max_length=self.config.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        vision_inputs = self.vision_processor(
            images=image,
            return_tensors="pt"
        )
        clip_inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.config.clip_max_text_length
        )
        clip_inputs.pop("token_type_ids", None)
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': vision_inputs['pixel_values'].squeeze(0),

            'clip_input_ids': clip_inputs['input_ids'].squeeze(0),
            'clip_attention_mask': clip_inputs['attention_mask'].squeeze(0),
            'clip_pixel_values': clip_inputs['pixel_values'].squeeze(0),

            'label': torch.tensor(label, dtype=torch.float32),
            'text': text,
            'image_path': image_path,
            'category': category,
        }


def raw_collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'clip_input_ids': torch.stack([item['clip_input_ids'] for item in batch]),
        'clip_attention_mask': torch.stack([item['clip_attention_mask'] for item in batch]),
        'clip_pixel_values': torch.stack([item['clip_pixel_values'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'text': [item['text'] for item in batch],
        'image_path': [item['image_path'] for item in batch],
        'category': [item['category'] for item in batch],
    }


def create_raw_data_loaders(data_dir, image_dir, config: Config):
    full_image_dir = os.path.join(data_dir, image_dir)
    loaders = []
    for split in ('train', 'val', 'test'):
        csv_path = os.path.join(data_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            loaders.append(None)
            continue
        dataset = RawImageTextDataset(csv_path, full_image_dir, config, split=split)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                collate_fn=raw_collate_fn,
            )
        )
    return tuple(loaders)


def load_backbones(config):
    print("正在加载预训练骨干网络...")
    text_model = AutoModel.from_pretrained(config.text_model_name)
    vision_model = AutoModel.from_pretrained(config.vision_model_name)
    if 'chinese' in config.clip_model_name.lower():
        clip_model = ChineseCLIPModel.from_pretrained(config.clip_model_name)
    else:
        clip_model = CLIPModel.from_pretrained(config.clip_model_name)
    for model in [text_model, vision_model, clip_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return text_model, vision_model, clip_model


def extract_and_save_features(loader, models, device, output_path, config):
    text_model, vision_model, clip_model = models
    extracted_data = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"提取特征至 {os.path.basename(output_path)}")):
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                text_out = text_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                text_hidden = text_out.last_hidden_state
                vision_out = vision_model( pixel_values=inputs['pixel_values'])
                vision_hidden = vision_out.last_hidden_state
                clip_outputs = clip_model(
                    input_ids=inputs['clip_input_ids'],
                    attention_mask=inputs['clip_attention_mask'],
                    pixel_values=inputs['clip_pixel_values'],
                    return_dict=True,
                    output_hidden_states=True
                )
                clip_text_global = clip_outputs.text_embeds
                clip_image_global = clip_outputs.image_embeds
                clip_text_local = clip_outputs.text_model_output.last_hidden_state
                clip_vision_local = clip_outputs.vision_model_output.last_hidden_state
            text_hidden = text_hidden.cpu().half()
            vision_hidden = vision_hidden.cpu().half()
            clip_text_global = clip_text_global.cpu().half()
            clip_image_global = clip_image_global.cpu().half()
            clip_text_local = clip_text_local.cpu().half()
            clip_vision_local = clip_vision_local.cpu().half()
            attention_masks = inputs['attention_mask'].cpu()
            clip_attention_masks = inputs['clip_attention_mask'].cpu()
            labels = inputs['label'].cpu()
            batch_size = labels.size(0)
            for i in range(batch_size):
                sample_dict = {
                    'text_hidden': text_hidden[i],
                    'vision_hidden': vision_hidden[i],
                    'clip_text_global': clip_text_global[i],
                    'clip_image_global': clip_image_global[i],
                    'clip_text_local': clip_text_local[i],
                    'clip_vision_local': clip_vision_local[i],
                    'attention_mask': attention_masks[i],
                    'clip_attention_mask': clip_attention_masks[i],
                    'label': labels[i].item(),
                    'text': batch['text'][i],
                    'image_path': batch['image_path'][i],
                    'category': batch['category'][i],
                }
                extracted_data.append(sample_dict)
    torch.save(extracted_data, output_path)
    print(f"成功保存 {len(extracted_data)} 条特征至 {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract frozen backbone features for GAHR.')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--image_dir', default='images')
    parser.add_argument('--feature_dir', default='features')
    parser.add_argument('--language', choices=['zh', 'en'], default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(language=args.language)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        config.batch_size = args.batch_size
    device = torch.device(config.device)
    os.makedirs(args.feature_dir, exist_ok=True)
    train_loader, val_loader, test_loader = create_raw_data_loaders(args.data_dir, args.image_dir, config)
    if all(loader is None for loader in (train_loader, val_loader, test_loader)):
        raise FileNotFoundError(f'No split CSV files found in {args.data_dir}.')
    models = load_backbones(config)
    for model in models:
        model.to(device)
    if train_loader is not None:
        extract_and_save_features(train_loader, models, device, os.path.join(args.feature_dir, 'train_features.pt'), config)
    if val_loader is not None:
        extract_and_save_features(val_loader, models, device, os.path.join(args.feature_dir, 'val_features.pt'), config)
    if test_loader is not None:
        extract_and_save_features(test_loader, models, device, os.path.join(args.feature_dir, 'test_features.pt'), config)
    print('Feature extraction complete.')


if __name__ == '__main__':
    main()
