
# from comet_ml import start

from utils import set_seeds


def main(opts):
    set_seeds(opts.seed)


if __name__ == "__main__":
    from cmd_args import parse_args
    args = parse_args()
    try:
        main(args)
    except Exception:
        import ipdb
        ipdb.post_mortem()
