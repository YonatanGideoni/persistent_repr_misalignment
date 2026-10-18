import os

from small_scale_failures.clip_failure.clip_model_utils import CLIPDualEncoderConfig
from small_scale_failures.model_utils import AdapterConfig
from small_scale_failures.training_utils import get_base_parser


def get_parser():
    parser = get_base_parser("Train CLIP model with adapters", default_bs=256)
    parser.add_argument("--imagenet_dir", type=str, default=os.environ.get('IMAGENET_DIR'),
                        help="Directory containing ImageNet (used for zero-shot ImageNet eval), defaults to $IMAGENET_DIR")
    parser.add_argument("--vis_pooling", type=str, default="mean",
                        help="Pooling method for visual adapter")
    parser.add_argument("--lang_pooling", type=str, default="mean",
                        help="Pooling method for language adapter")
    return parser


def get_config(args):
    vis_adapter_config = AdapterConfig(adapter_type=args.adapter_type, pooling=args.vis_pooling)
    lang_adapter_config = AdapterConfig(adapter_type=args.adapter_type, pooling=args.lang_pooling)

    config = CLIPDualEncoderConfig(
        vis_adapter_config=vis_adapter_config,
        lang_adapter_config=lang_adapter_config,
        vis_enc=args.vis_enc,
        lang_enc=args.lang_enc
    )

    return config
