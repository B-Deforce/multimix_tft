import pandas as pd
from custom_dataset import TimeSeriesDataLoader
import pytorch_lightning as pl
from multimix_tft import TemporalFusionTransformer as tft

# argparser
import argparse
import yaml


def create_ckpt_name_and_logger(config):
    # Extract columns, omitting prefix and filtering
    cols = ""
    cols = cols.join(
        i.replace("vmc_", "")
        for i in config["model_params"]["historical_real_cols"]
        if "vmc" in i
    )
    target = config["model_params"]["target_col"][0].replace("vmc_", "")
    # Calculate time gap
    time_gap = config["model_params"]["time_gap"] * 4
    ckpt_name = f"MultiMix_{cols}_{target}_{time_gap}"

    # create ckpt logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",  # Base directory for logs
        name=ckpt_name,  # Top-level directory name
        version=None,  # Auto-incrementing version number
    )

    # create csv logger
    csv_logger = pl.loggers.CSVLogger(
        save_dir="lightning_logs",
        name=ckpt_name,
        version=tb_logger.version,  # Use the same version directory as the TensorBoard logger
    )
    loggers = [tb_logger, csv_logger]
    return ckpt_name, loggers


def main(args):
    # set global seed
    pl.seed_everything(96)
    # load config
    with open(args.config_path) as cfg:
        config = yaml.safe_load(cfg)

    # q = input(
    # f"Time gap is set to {config['model_params']['time_gap']} unit."
    # + f"\nTarget is set to {config['model_params']['target_col']}."
    # + f"\nand historical variables are set to {config['model_params']['historical_real_cols']}."
    # + "\nDo you want to continue? (y/n): "
    # )
    # if q.lower() != "y":
    # print("Exiting...")
    # exit()

    m_config = config["model_params"]
    t_params = config["train_params"]
    # load data
    train_df = pd.read_parquet(config["train_path"])
    val_df = pd.read_parquet(config["val_path"])
    test_df = pd.read_parquet(config["test_path"])
    # print shapes
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
        target=m_config["target_col"],
        window_size=m_config["window_size"],
        group_ids=m_config["group_col"],
        batch_size=m_config["batch_size"],
        time_idx=m_config["time_col"],
        mixed_only=m_config["mixed_only"],
        time_gap=m_config["time_gap"],
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Initialize model
    model = tft(
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
    )

    # name custom checkpoint
    model_name, loggers = create_ckpt_name_and_logger(config)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mse_loss1",
        filename=model_name + "-{epoch:02d}-{val_mse_loss1:.2f}",
        save_top_k=1,
        save_last=True,
        mode="min",
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

    args = parser.parse_args()
    main(args)
