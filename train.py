import src.utils as utils
from src import build_criterion, build_dataloader, build_model, build_optimizer
from src.trainer import ClassificationTrainer


def main(args):
    dataloader_config = args.get("dataloader", {})

    train_dataloader = build_dataloader(dataloader_config.get("train", {}))
    validation_dataloader = build_dataloader(dataloader_config.get("val", {}))

    model = build_model(args.get("model", {}))
    optimizer = build_optimizer(model.parameters(), args.get("optim", {}))

    criterion = build_criterion(args.get("loss", {}))

    trainer = ClassificationTrainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        validation_dataloader,
        train_config=args.get("train", {}),
    )
    trainer.train()


if __name__ == "__main__":
    args = utils.parse_args()
    utils.setup_project(args)

    main(args)
