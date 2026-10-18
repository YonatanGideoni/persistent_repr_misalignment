from small_scale_failures.generation_failure.gen_model_utils import GenerativeVLMConfig
from small_scale_failures.model_utils import AdapterConfig
from small_scale_failures.training_utils import get_base_parser


def get_parser():
    parser = get_base_parser("Train an I2T VLM model", default_bs=64)
    parser.add_argument("--num_tokens", type=int, default=200,
                        help="Number of learnable tokens for the attention adapter")
    return parser


def get_config(args):
    adapter_config = AdapterConfig(adapter_type=args.adapter_type, adapter_kwargs={'num_tokens': args.num_tokens},
                                   pooling='none')

    config = GenerativeVLMConfig(
        vision_encoder=args.vis_enc,
        text_decoder=args.lang_enc,
        adapter_config=adapter_config
    )

    return config
