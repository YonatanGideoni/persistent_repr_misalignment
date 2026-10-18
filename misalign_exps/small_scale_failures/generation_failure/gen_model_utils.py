import torch
from transformers import GenerationConfig, PreTrainedModel, PretrainedConfig, ViTModel, GPT2LMHeadModel

from constants import device
from small_scale_failures.model_utils import AdapterConfig, ADAPTERS


class GenerativeVLMConfig(PretrainedConfig):
    """Configuration class for GenerativeVLM."""

    model_type = "generative_vlm"
    is_composition = True

    def __init__(
            self,
            vision_encoder="google/vit-base-patch16-224",
            text_decoder="gpt2",
            adapter_config: AdapterConfig = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder_model = vision_encoder
        self.text_decoder_model = text_decoder

        # Initialize adapter config if not provided
        if adapter_config is None:
            adapter_config = AdapterConfig(
                adapter_type='linear',
                pooling='none'
            )
        self.adapter_config = adapter_config

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict['adapter_config'] = self.adapter_config.to_dict()
        return base_dict

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        adapter_dict = config_dict.pop('adapter_config', {})
        adapter_config = AdapterConfig.from_dict(adapter_dict)

        return cls(
            adapter_config=adapter_config,
            **config_dict
        )


class GenerativeVLM(PreTrainedModel):
    """
    A generative vision-language model that converts vision tokens into soft prompts
    for conditioning text generation.

    Architecture:
    1. Vision encoder (ViT) processes images
    2. Adapter projects vision tokens to text embedding space
    3. Projected tokens serve as soft prompts prepended to text decoder
    4. Text decoder (GPT-2) generates captions
    """

    config_class = GenerativeVLMConfig

    def __init__(self, config: GenerativeVLMConfig):
        super().__init__(config)
        self.config = config

        # Initialize vision encoder and text decoder
        self.vision_encoder = ViTModel.from_pretrained(config.vision_encoder_model)
        self.text_decoder = GPT2LMHeadModel.from_pretrained(config.text_decoder_model)

        # pad token is vocab+1
        self.text_decoder.resize_token_embeddings(self.text_decoder.config.vocab_size + 1)
        config.pad_token_id = self.text_decoder.config.vocab_size - 1

        self.generation_config = GenerationConfig.from_model_config(config)

        # Get hidden sizes
        vision_hidden_size = self.vision_encoder.config.hidden_size
        text_hidden_size = self.text_decoder.config.n_embd

        # Update adapter config dimensions
        config.adapter_config.input_dim = vision_hidden_size
        config.adapter_config.output_dim = text_hidden_size

        # Initialize vision-to-text adapter
        adapter_class = ADAPTERS[config.adapter_config.adapter_type]
        self.vision_to_text_adapter = adapter_class(config.adapter_config)

        # Freeze encoders by default
        self.vision_encoder.eval()
        self.text_decoder.eval()

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.text_decoder.parameters():
            param.requires_grad = False

    def get_soft_prompts(self, pixel_values):
        """Extract vision features and convert them to soft prompts."""
        # Move to same device as model
        pixel_values = pixel_values.to(device)

        # Encode images with ViT
        with torch.no_grad():
            self.vision_encoder.eval()
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                return_dict=True
            )

        # Get vision token sequence
        vision_tokens = vision_outputs.last_hidden_state

        # Project to text embedding space
        soft_prompts = self.vision_to_text_adapter(vision_tokens)

        return soft_prompts

    def forward(
            self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        """Forward pass for training and inference."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get soft prompts from vision
        soft_prompts = self.get_soft_prompts(pixel_values)
        batch_size = soft_prompts.shape[0]
        soft_prompt_length = soft_prompts.shape[1]

        # Handle input_ids and labels
        if input_ids is None and labels is not None:
            input_ids = labels

        # Get text embeddings if input_ids provided
        inputs_embeds = soft_prompts
        if input_ids is not None:
            text_embeds = self.text_decoder.transformer.wte(input_ids)
            # Concatenate soft prompts with text embeddings
            inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

        # Create attention mask if not provided
        if attention_mask is None and input_ids is not None:
            # Assuming pad_token_id is the last token in vocabulary
            pad_token_id = self.text_decoder.config.vocab_size - 1  # Special added token
            attention_mask = input_ids != pad_token_id

        # Prepare labels for loss computation
        extended_labels = None
        if labels is not None:
            # Create ignore labels (-100) for soft prompt section
            ignore_labels = torch.full(
                (batch_size, soft_prompt_length),
                -100,
                dtype=labels.dtype,
                device=device
            )

            # For text tokens, set padding tokens to -100
            if attention_mask is not None:
                labels = torch.where(attention_mask, labels, -100)

            # Concatenate to match input length
            extended_labels = torch.cat([ignore_labels, labels], dim=1)

        # Extend attention mask to include soft prompts
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(
                (batch_size, soft_prompt_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            extended_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)
        else:
            # Full attention mask if no input_ids
            n_tokens = inputs_embeds.shape[1]
            extended_attention_mask = torch.ones(
                (batch_size, n_tokens),
                dtype=torch.long,
                device=inputs_embeds.device
            )

        # Forward through decoder
        self.text_decoder.eval()
        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return outputs

    def generate(
            self,
            pixel_values,
            input_ids=None,
            attention_mask=None,
            **generate_kwargs
    ):
        """Generate text given images."""
        batch_size = pixel_values.shape[0]

        # Get soft prompts from vision
        soft_prompts = self.get_soft_prompts(pixel_values)

        # Prepare inputs
        inputs_embeds = soft_prompts
        if input_ids is not None:
            text_embeds = self.text_decoder.transformer.wte(input_ids)
            inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

        # Update attention mask
        combined_mask = None
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(
                (batch_size, soft_prompts.shape[1]),
                device=attention_mask.device
            )
            combined_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Set default generation parameters
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 64

        return self.text_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            **generate_kwargs
        )
