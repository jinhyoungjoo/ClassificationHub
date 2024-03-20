import src.utils as utils
from src.dataloader import build_dataloader


def main(args):
    dataloader_config = args.get("dataloader", {})

    train_dataloader = build_dataloader(dataloader_config.get("train", {}))
    validation_dataloader = build_dataloader(dataloader_config.get("val", {}))




if __name__ == "__main__":
    args = utils.parse_args()
    utils.setup_project(args)

    main(args)
