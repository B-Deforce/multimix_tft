# argparser
import argparse

import pandas as pd
import pytorch_lightning as pl
import yaml

from dataloader.custom_dataset import TimeSeriesDataLoader
from multimix.multimix_tft import MultiMixTFT
from multimix.multimix_tft_classifier import MultiMixTFTC
from multimix.transfer_multimix_tft import MultiMixTFTT
from utils import create_ckpt_name_and_logger


def main(args):
    """Main function to train the MultiMix model.

    Args:
        args: Command line arguments containing configuration paths and training parameters.
    """
    # set global seed
    pl.seed_everything(96)

    # load config
    with open(args.config_path) as cfg:
        config = yaml.safe_load(cfg)

    q = input(
        f"Time gap is set to {config['model_params']['time_gap']} unit."
        + f"\nPrimary Target is set to {config['model_params']['primary_target_col']}."
        + f"\nand historical variables are set to {config['model_params']['historical_real_cols']}."
        + "\nDo you want to continue? (y/n): "
    )
    if q.lower() != "y":
        print("Exiting...")
        exit()

    m_config = config["model_params"]

    # load data
    train_df = pd.read_parquet(config["train_path"])
    val_df = pd.read_parquet(config["val_path"])
    test_df = pd.read_parquet(config["test_path"])
    print("Train shape: ", train_df.shape)
    print("Val shape: ", val_df.shape)
    print("Test shape: ", test_df.shape)

    # Initialize data module
    data_module = TimeSeriesDataLoader(
        train=train_df,
        val=val_df,
        test=test_df,
        static_real_cols=m_config["static_real_cols"],
        static_cat_cols=m_config["static_cat_cols"],
        historical_real_cols=m_config["historical_real_cols"],
        historical_cat_cols=m_config["historical_cat_cols"],
        known_real_cols=m_config["known_real_cols"],
        known_cat_cols=m_config["known_cat_cols"],
        primary_target=m_config["primary_target_col"],
        mf_target=m_config["mf_target_col"],
        classification_target=m_config.get("binary_target_col", None),
        window_size=m_config["window_size"],
        group_ids=m_config["group_col"],
        batch_size=m_config["batch_size"],
        time_idx=m_config["time_col"],
        time_gap=m_config["time_gap"],
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    if m_config.get("binary_target_col") is not None:
        assert args.task == "classification", (
            "Binary target column is only applicable for classification task."
        )

    # Initialize model
    if args.finetune == "partial":
        model = MultiMixTFTT(
            pretrained_model_path=m_config["pretrained_model_path"],
            output_size=m_config["output_size"],
            quantiles=m_config["quantiles"],
        )
    elif args.finetune == "full":
        model = MultiMixTFT.load_from_checkpoint(
            checkpoint_path=m_config["pretrained_model_path"],
        )
    elif args.task == "regression":
        model = MultiMixTFT(
            hidden_layer_size=m_config["hidden_layer_size"],
            static_categorical_sizes=data_module.static_cat_sizes,
            historical_categorical_sizes=data_module.historical_cat_sizes,
            static_reals=m_config["static_real_cols"],
            historical_reals=m_config["historical_real_cols"],
            known_categoricals=m_config["known_cat_cols"],
            known_reals=m_config["known_real_cols"],
            dropout_rate=m_config["dropout_rate"],
            num_heads=m_config["num_heads"],
            output_size=m_config["output_size"],
            quantiles=m_config["quantiles"],
            window_size=data_module.window_size,
        )
    elif args.task == "classification":
        model = MultiMixTFTC(
            hidden_layer_size=m_config["hidden_layer_size"],
            static_categorical_sizes=data_module.static_cat_sizes,
            historical_categorical_sizes=data_module.historical_cat_sizes,
            static_reals=m_config["static_real_cols"],
            historical_reals=m_config["historical_real_cols"],
            known_categoricals=m_config["known_cat_cols"],
            known_reals=m_config["known_real_cols"],
            dropout_rate=m_config["dropout_rate"],
            num_heads=m_config["num_heads"],
            output_size=m_config["output_size"],
            quantiles=m_config["quantiles"],
            window_size=data_module.window_size,
            weights=data_module.weights,
        )

    # create loggers and checkpoint callbacks
    loggers = create_ckpt_name_and_logger(args.ckpt_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mse_loss1" if args.task == "regression" else "val_ap",
        filename="MultiMix_" + args.ckpt_name + "-{epoch:02d}-{val_mse_loss1:.2f}"
        if args.task == "regression"
        else "MultiMix_" + args.ckpt_name + "-{epoch:02d}-{val_ap:.2f}",
        save_top_k=1,
        save_last=True,
        mode="min" if args.task == "regression" else "max",
    )
    # init trainer
    trainer = pl.Trainer(
        logger=loggers,
        max_epochs=config["train_params"]["epochs"],
        callbacks=[checkpoint_callback],
    )
    # fit trainer
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiMix.")
    parser.add_argument("--config_path", type=str)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path to resume training, if any.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="MultiMix",
        help="Checkpoint name to save the model.",
    )
    parser.add_argument(
        "--finetune",
        choices=["full", "partial"],
        default=None,
        help="Finetune the model.",
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Task type.",
    )
    args = parser.parse_args()
    main(args)
