import json

import pytorch_lightning as pl
import yaml


def create_ckpt_name_and_logger(ckpt_name):
    """Create a checkpoint name and logger for PyTorch Lightning.
    Args:
        ckpt_name (str): The name for the checkpoint.
    Returns:
        loggers (list): A list containing the TensorBoard and CSV loggers.
    """

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name="MultiMix_" + ckpt_name,
    )

    csv_logger = pl.loggers.CSVLogger(
        save_dir="lightning_logs",
        name="MultiMix_" + ckpt_name,
        version=tb_logger.version,  # Use the same version directory as the TensorBoard logger
    )
    loggers = [tb_logger, csv_logger]
    return loggers


def get_config(config_path):
    with open(config_path) as cfg:
        config = yaml.safe_load(cfg)
    return config


def read_json(json_path):
    with open(json_path, "r") as file:
        f = json.load(file)
    return f
