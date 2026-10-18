import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer


class AdapterConfig(PretrainedConfig):
    model_type = "clip_adapters"

    def __init__(self,
                 input_dim: int = None,
                 output_dim: int = None,
                 adapter_type: str = 'linear',
                 pooling: str = 'mean',
                 adapter_kwargs: dict = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapter_type = adapter_type
        self.pooling = pooling
        self.adapter_kwargs = adapter_kwargs if adapter_kwargs is not None else {}


class Adapter(PreTrainedModel):
    """
    Base class for adapters. Should be subclassed for specific adapter types.
    """

    config_class = AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

    def pool(self, x, attention_mask=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        if self.config.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).type_as(x)
            return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.config.pooling == 'max':
            return x.masked_fill(~attention_mask.unsqueeze(-1), float('-inf')).max(1).values
        if self.config.pooling == 'cls':
            return x[:, 0, :]
        if self.config.pooling == 'none':
            return x
        raise NotImplementedError

    def forward(self, x, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class LinearAdapter(Adapter):
    """
    A simple linear adapter that projects tokens from input dim to some output dim.
    """

    config_class = AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x, **kwargs):
        x = self.pool(x, **kwargs)
        return self.linear(x)


class MLPAdapter(Adapter):
    """
    A multi-layer perceptron adapter with configurable hidden dimensions.
    """

    config_class = AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__(config)

        # Get MLP-specific parameters from adapter_kwargs
        hidden_dim = config.adapter_kwargs.get('hidden_dim', config.input_dim)
        num_layers = config.adapter_kwargs.get('num_layers', 2)

        # Build MLP layers
        layers = []
        in_dim = config.input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim

        # Final projection layer
        layers.append(nn.Linear(in_dim, config.output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.pool(x, **kwargs)
        return self.mlp(x)


class AttentionAdapter(Adapter):
    """
    An adapter that uses learnable global soft tokens with cross attention between them and the input tokens.
    """

    config_class = AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__(config)

        # Get attention-specific parameters
        num_heads = config.adapter_kwargs.get('num_heads', 8)
        self.num_tokens = config.adapter_kwargs.get('num_tokens', 1)

        # Learnable global tokens
        self.global_tokens = nn.Parameter(
            torch.randn(self.num_tokens, config.input_dim)
        )

        # Multi-head cross attention where global tokens attend to input tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.input_dim)

        # Final projection
        self.output_proj = nn.Linear(config.input_dim, config.output_dim) if self.num_tokens > 1 else nn.Identity()

    def forward(self, x, attention_mask=None, **kwargs):
        """
        Args:
            x: Input tokens [batch_size, seq_len, input_dim]
            attention_mask: Mask for input tokens [batch_size, seq_len]
        """
        batch_size = x.shape[0]

        # Expand global tokens for each sample in the batch
        global_tokens = self.global_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Create attention mask for cross-attention
        key_padding_mask = None
        if attention_mask is not None:
            # Convert boolean mask to float mask expected by attention
            key_padding_mask = ~attention_mask.bool()  # Invert: True where padding

        # Cross attention: global tokens attend to input sequence
        attended_global, _ = self.cross_attention(
            query=global_tokens,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )

        # Residual connection and layer norm
        attended_global = self.layer_norm(global_tokens + attended_global)

        # Project to output dimension
        output = self.output_proj(attended_global)

        if self.num_tokens == 1:
            output = output.squeeze(1)

        return output


ADAPTERS = {
    'linear': LinearAdapter,
    'mlp': MLPAdapter,
    'attention': AttentionAdapter
}


def get_tokenizer(lang_model: str):
    tokenizer = AutoTokenizer.from_pretrained(lang_model, use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
