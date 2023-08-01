import argparse
from bispectral_networks.trainer import run_trainer

from mk_mlutils.utils import torchutils


def run_wrapper():
    run_trainer(
        master_config=master_config,
        logger_config=logger_config,
        device=args.device,
        n_examples=args.n_examples,
        epochs=args.epochs,
        seed=args.seed
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Name of .py config file with no extension.",
        default="translation_experiment",
    )
    parser.add_argument("--device", type=int, help="device to run on, -1 for cpu", default=0)
    parser.add_argument(
        "--n_examples", type=int, help="number of data examples", default=21504
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    if args.device == -1:
        args.device = 'cpu'
            
    print(f"Running {args.config} on device {args.device}...")
    exec(f"from configs.{args.config} import master_config, logger_config")

    run_wrapper()
