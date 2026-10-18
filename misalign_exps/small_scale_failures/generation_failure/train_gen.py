import os

from small_scale_failures.generation_failure.gen_model_utils import GenerativeVLM
from small_scale_failures.generation_failure.misc_utils import get_parser, get_config
from small_scale_failures.generation_failure.training_utils import train_gen
from small_scale_failures.training_utils import train_main


def main():
    parser = get_parser()

    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, f'excl_conc_{args.exclude_concepts}', args.adapter_type)

    train_main(args, get_config, GenerativeVLM, train_gen)


if __name__ == '__main__':
    main()
