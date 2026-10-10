import torch


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Input and feature extraction
    max_text_length = 256
    image_size = 224
    num_workers = 0
    pin_memory = True

    # Backbone profiles
    language = 'zh'
    text_model_name = 'hfl/chinese-roberta-wwm-ext'
    clip_model_name = 'OFA-Sys/chinese-clip-vit-large-patch14'
    clip_max_text_length = 128
    english_text_model_name = 'roberta-base'
    english_clip_model_name = 'openai/clip-vit-large-patch14'
    english_clip_max_text_length = 77
    vision_model_name = 'facebook/dinov2-base'

    # Feature dimensions
    text_hidden_dim = 768
    vision_hidden_dim = 768
    clip_embed_dim = 768
    clip_text_local_dim = 768
    clip_vision_local_dim = 1024
    unified_dim = 512
    num_attention_heads = 8
    attention_temperature = 0.5

    # Optimization
    output_root = 'outputs'
    best_model_path = 'outputs/model/best_model.pth'
    num_epochs = 50
    batch_size = 64
    accumulation_steps = 1
    learning_rate = 5e-5
    weight_decay = 0.1
    gradient_clip_norm = 1.0
    patience = 5
    min_delta = 1e-4
    drop_prob = 0.1
    generativity_loss_weight = 0.2

    def __init__(self, language=None):
        self.set_language(language or type(self).language)

    def set_language(self, language):
        language = str(language).lower()
        if language not in {'zh', 'en'}:
            raise ValueError("language must be 'zh' or 'en'")

        self.language = language
        if language == 'zh':
            self.text_model_name = 'hfl/chinese-roberta-wwm-ext'
            self.clip_model_name = 'OFA-Sys/chinese-clip-vit-large-patch14'
            self.clip_max_text_length = 128
        else:
            self.text_model_name = self.english_text_model_name
            self.clip_model_name = self.english_clip_model_name
            self.clip_max_text_length = self.english_clip_max_text_length
