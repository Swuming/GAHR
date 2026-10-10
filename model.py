import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, dim, temperature=0.5):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, query_features, key_features, value_features, mask=None):
        query = self.query(query_features)
        key = self.key(key_features)
        value = self.value(value_features)
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = scores / (self.temperature * (self.dim ** 0.5))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)


class GlobalInteraction(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"unified_dim={dim} must be divisible by num_heads={num_heads}")
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.multiply_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, text_response, image_response):
        response_sequence = torch.stack([text_response, image_response], dim=1)
        updated, _ = self.self_attention(
            response_sequence,
            response_sequence,
            response_sequence,
            need_weights=False,
        )
        updated_text, updated_image = updated[:, 0], updated[:, 1]
        s_global = 0.5 * (updated_text + updated_image)
        r_global = self.multiply_projection(text_response * image_response)
        gate = self.fusion_gate(torch.cat([s_global, r_global, s_global * r_global], dim=-1))
        v_global = self.layer_norm(gate * s_global + (1.0 - gate) * r_global)
        return v_global


class MultiModalConsistencyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.unified_dim
        self.proj_text_global = self._build_projection(config.text_hidden_dim)
        self.proj_text_local = self._build_projection(config.text_hidden_dim)
        self.proj_image_global = self._build_projection(config.vision_hidden_dim)
        self.proj_image_local = self._build_projection(config.vision_hidden_dim)
        self.proj_clip_text_global = self._build_projection(config.clip_embed_dim)
        self.proj_clip_image_global = self._build_projection(config.clip_embed_dim)
        self.proj_clip_text_local = self._build_projection(config.clip_text_local_dim)
        self.proj_clip_image_local = self._build_projection(config.clip_vision_local_dim)
        attention_temperature = getattr(config, 'attention_temperature', 0.5)
        self.text_to_image_attention = CrossModalAttention(self.dim, temperature=attention_temperature)
        self.image_to_text_attention = CrossModalAttention(self.dim, temperature=attention_temperature)
        self.global_interaction = GlobalInteraction(self.dim, num_heads=getattr(config, 'num_attention_heads', 8),)
        self.global_state_net = nn.Sequential(
            nn.Linear(5 * self.dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.drop_prob),
            nn.Linear(256, self.dim),
        )
        self.global_prior_head = nn.Linear(self.dim, 1)
        self.text_query = nn.Linear(self.dim, self.dim)
        self.image_query = nn.Linear(self.dim, self.dim)
        self.text_evidence_projection = self._build_projection(4 * self.dim)
        self.image_evidence_projection = self._build_projection(4 * self.dim)
        self.consistency_evidence_projection = self._build_projection(4 * self.dim)
        self.local_correction_head = nn.Sequential(
            nn.Linear(3 * self.dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.drop_prob),
            nn.Linear(512, 1),
        )
        self.correction_gate = nn.Sequential(
            nn.Linear(4 * self.dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.drop_prob),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.generativity_projection = self._build_projection(4 * self.dim)
        self.text_generativity_head = nn.Linear(self.dim, 1)
        self.image_generativity_head = nn.Linear(self.dim, 1)

    def _build_projection(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
        )

    @staticmethod
    def _masked_pool(features, query, mask=None):
        scores = torch.bmm(features, query.unsqueeze(-1)).squeeze(-1)
        scores = scores / (features.size(-1) ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return pooled

    @staticmethod
    def _relation(left, right):
        return torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)

    def _global_reasoning(self, text_global, text_local, image_global, image_local, clip_text_global, clip_image_global, text_mask,):
        image_response = self.text_to_image_attention(clip_text_global.unsqueeze(1), image_local, image_local)
        text_response = self.image_to_text_attention(clip_image_global.unsqueeze(1), text_local, text_local, mask=text_mask)
        text_response = text_response.squeeze(1)
        image_response = image_response.squeeze(1)
        global_interaction = self.global_interaction(text_response, image_response)
        global_state = self.global_state_net(torch.cat([global_interaction, text_global, image_global, clip_text_global, clip_image_global,], dim=-1,))
        global_logits = self.global_prior_head(global_state).squeeze(-1)
        return (
            global_state,
            global_logits,
            text_response,
            image_response,
        )

    def forward(self, text_hidden, vision_hidden, clip_text_global, clip_image_global, clip_text_local, clip_vision_local, attention_mask, clip_attention_mask):
        text_global = self.proj_text_global(text_hidden[:, 0, :])
        image_global = self.proj_image_global(vision_hidden[:, 0, :])
        clip_text_global = self.proj_clip_text_global(clip_text_global)
        clip_image_global = self.proj_clip_image_global(clip_image_global)
        text_local = self.proj_text_local(text_hidden[:, 1:, :])
        image_local = self.proj_image_local(vision_hidden[:, 1:, :])
        clip_text_local = self.proj_clip_text_local(clip_text_local[:, 1:, :])
        clip_image_local = self.proj_clip_image_local(clip_vision_local[:, 1:, :])
        text_mask = attention_mask[:, 1:]
        clip_text_mask = clip_attention_mask[:, 1:]
        (global_state, global_logits, text_response, image_response,) = self._global_reasoning(text_global, text_local, image_global, image_local, clip_text_global, clip_image_global, text_mask)
        text_query = self.text_query(text_response)
        image_query = self.image_query(image_response)
        text_backbone = self._masked_pool(text_local, text_query, text_mask)
        image_backbone = self._masked_pool(image_local, image_query)
        text_clip = self._masked_pool(clip_text_local, text_query, clip_text_mask)
        image_clip = self._masked_pool(clip_image_local, image_query)
        text_evidence = self.text_evidence_projection(self._relation(text_backbone, text_clip))
        image_evidence = self.image_evidence_projection(self._relation(image_backbone, image_clip))
        consistency_evidence = self.consistency_evidence_projection(self._relation(text_clip, image_clip))
        local_state = torch.cat([text_evidence, image_evidence, consistency_evidence], dim=-1)
        local_logits = self.local_correction_head(local_state).squeeze(-1)
        correction_gate = self.correction_gate(torch.cat([global_state, text_evidence, image_evidence, consistency_evidence], dim=-1,)).squeeze(-1)
        final_logits = global_logits + correction_gate * local_logits
        generativity_state = self.generativity_projection(torch.cat([global_state, local_state], dim=-1))
        text_generativity_logits = self.text_generativity_head(generativity_state).squeeze(-1)
        image_generativity_logits = self.image_generativity_head(generativity_state).squeeze(-1)
        return {
            'logits': final_logits,
            'prediction': torch.sigmoid(final_logits),
            'text_generativity_logits': text_generativity_logits,
            'image_generativity_logits': image_generativity_logits,
        }


class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_a = float(getattr(config, 'generativity_loss_weight', 0.2))

    @staticmethod
    def _zero_like(reference):
        return reference.sum() * 0.0

    def forward(self, outputs, labels, batch=None):
        veracity_loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels)
        generativity_loss = self._zero_like(outputs['logits'])
        if batch is not None:
            mask = batch.get('generativity_mask')
            text_labels = batch.get('text_generativity_label')
            image_labels = batch.get('image_generativity_label')
            text_logits = outputs.get('text_generativity_logits')
            image_logits = outputs.get('image_generativity_logits')
            if (
                mask is not None
                and text_labels is not None
                and image_labels is not None
                and text_logits is not None
                and image_logits is not None
            ):
                valid = mask > 0.5
                if valid.any():
                    text_loss = F.binary_cross_entropy_with_logits(text_logits[valid], text_labels[valid], reduction='none')
                    image_loss = F.binary_cross_entropy_with_logits(image_logits[valid], image_labels[valid], reduction='none')
                    generativity_loss = (text_loss + image_loss).mean()
        total_loss = veracity_loss + self.lambda_a * generativity_loss
        return {
            'total_loss': total_loss,
            'veracity_loss': veracity_loss,
            'generativity_loss': generativity_loss,
        }
