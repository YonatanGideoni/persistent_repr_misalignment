import os

from small_scale_failures.clip_failure.clip_model_utils import CLIPDualEncoderModel
from small_scale_failures.clip_failure.misc_utils import get_parser, get_config
from small_scale_failures.clip_failure.training_utils import train_clip
from small_scale_failures.training_utils import train_main


def main():
    parser = get_parser()

    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, f'excl_conc_{args.exclude_concepts}', args.adapter_type, args.vis_pooling)

    train_main(args, get_config, CLIPDualEncoderModel, train_clip, append_eos=False)


if __name__ == '__main__':
    main()
