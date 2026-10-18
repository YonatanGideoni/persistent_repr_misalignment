import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, ViTModel, GPT2Model, AutoConfig

from constants import device
from small_scale_failures.model_utils import AdapterConfig, ADAPTERS


class CLIPDualEncoderConfig(PretrainedConfig):
    """Config for CLIPDualEncoderModel."""

    model_type = "clip_dual_encoder"
    is_composition = True

    def __init__(
            self,
            vis_adapter_config: AdapterConfig = AdapterConfig(),
            lang_adapter_config: AdapterConfig = AdapterConfig(),
            proj_dim: int = 512,
            vis_enc="google/vit-base-patch16-224",
            lang_enc="gpt2",
            temperature=0.07,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.vis_enc = vis_enc
        self.lang_enc = lang_enc
        self.temperature = temperature

        vis_adapter_config.input_dim = AutoConfig.from_pretrained(vis_enc).hidden_size
        lang_adapter_config.input_dim = AutoConfig.from_pretrained(lang_enc).hidden_size
        vis_adapter_config.output_dim = proj_dim
        lang_adapter_config.output_dim = proj_dim

        self.vis_adapter_config = vis_adapter_config
        self.lang_adapter_config = lang_adapter_config

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict['vis_adapter_config'] = self.vis_adapter_config.to_dict()
        base_dict['lang_adapter_config'] = self.lang_adapter_config.to_dict()
        return base_dict

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        vis_adapter_dict = config_dict.pop('vis_adapter_config', {})
        lang_adapter_dict = config_dict.pop('lang_adapter_config', {})

        vis_adapter_config = AdapterConfig.from_dict(vis_adapter_dict)
        lang_adapter_config = AdapterConfig.from_dict(lang_adapter_dict)

        config = cls(
            vis_adapter_config=vis_adapter_config,
            lang_adapter_config=lang_adapter_config,
            **config_dict
        )
        if return_unused_kwargs:
            return config, config_dict
        return config


class CLIPDualEncoderModel(PreTrainedModel):
    """
    A CLIP-style model that takes (typically frozen) text and image encoders and finetunes their final embeddings to
    get CLIP-like representations.
    """

    config_class = CLIPDualEncoderConfig

    def __init__(self, config: CLIPDualEncoderConfig):
        super().__init__(config)
        self.config = config

        # Initialize encoders
        self.vision_encoder = ViTModel.from_pretrained(config.vis_enc)
        self.language_encoder = GPT2Model.from_pretrained(config.lang_enc)

        # pad token is vocab+1
        self.language_encoder.resize_token_embeddings(self.language_encoder.config.vocab_size + 1)

        # Projection layer
        self.vis_adapter = ADAPTERS[config.vis_adapter_config.adapter_type](config.vis_adapter_config)
        self.lang_adapter = ADAPTERS[config.lang_adapter_config.adapter_type](config.lang_adapter_config)

        modules_to_freeze = [self.vision_encoder, self.language_encoder]
        for module in modules_to_freeze:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def get_vision_features(self, pixel_values):
        pixel_values = pixel_values.to(device)

        self.vision_encoder.eval()

        with torch.no_grad():
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                return_dict=True,
            )

        vision_features = vision_outputs.last_hidden_state

        # Project to shared space
        vision_embeddings = self.vis_adapter(vision_features)

        return vision_embeddings

    def get_text_features(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = input_ids.ne(self.language_encoder.config.vocab_size - 1)  # ignore pad token

        self.language_encoder.eval()

        with torch.no_grad():
            outputs = self.language_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # Get hidden states from the last layer
        text_features = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]

        # Project to shared space, attention mask is important for pooling (ignore pad tokens)
        text_embeddings = self.lang_adapter(text_features, attention_mask=attention_mask)

        return text_embeddings

    def compute_loss_and_logits(self, vision_embeddings, text_embeddings):
        """
        Compute CLIP-style contrastive loss and similarity logits.
        Automatically detects duplicate vision embeddings to handle recurring images.

        Args:
            vision_embeddings: [batch_size, hidden_dim]
            text_embeddings: [batch_size, hidden_dim]

        Returns:
            loss: scalar loss value
            logits: [batch_size, batch_size] similarity matrix
        """
        # Assert embeddings are already pooled
        assert vision_embeddings.dim() == 2, f"Expected 2D vision embeddings, got {vision_embeddings.dim()}D"
        assert text_embeddings.dim() == 2, f"Expected 2D text embeddings, got {text_embeddings.dim()}D"

        # Normalize embeddings
        vision_embeddings = F.normalize(vision_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Compute similarity matrix between vision and text
        # logits[i,j] = similarity between image i and text j
        logits = torch.matmul(vision_embeddings, text_embeddings.t())

        # Scale by temperature
        temperature = self.config.temperature
        logits /= temperature

        # Compute loss
        # Detect duplicate vision embeddings using torch.isclose, compare all pairs using broadcasting
        labels = torch.isclose(
            vision_embeddings.unsqueeze(1),  # [batch, 1, hidden_dim]
            vision_embeddings.unsqueeze(0),  # [1, batch, hidden_dim]
        ).all(dim=-1).float()  # [batch, batch]

        # Convert labels to probabilities for each row/col
        # This handles cases where an image/caption has multiple valid captions/images
        image_to_text_probs = labels / labels.sum(dim=1, keepdim=True).clamp(min=1e-8)
        text_to_image_probs = labels.t() / labels.sum(dim=0, keepdim=True).clamp(min=1e-8)

        image_to_text_loss = -torch.sum(image_to_text_probs * F.log_softmax(logits, dim=1), dim=1).mean()
        text_to_image_loss = -torch.sum(text_to_image_probs * F.log_softmax(logits.t(), dim=1), dim=1).mean()

        # Total loss is the average of both directions
        loss = (image_to_text_loss + text_to_image_loss) / 2

        return loss, logits

    def forward(
            self,
            pixel_values,
            input_ids=None,
            attention_mask=None,
            labels=None,
            **kwargs
    ):
        if input_ids is None and labels is not None:
            input_ids = labels

        vision_embeddings = self.get_vision_features(pixel_values)
        text_embeddings = self.get_text_features(input_ids, attention_mask)

        loss, logits = self.compute_loss_and_logits(
            vision_embeddings=vision_embeddings,
            text_embeddings=text_embeddings
        )

        outputs = {
            'loss': loss,
            'logits': logits,
        }

        # Return a dictionary compatible with transformers
        return outputs
