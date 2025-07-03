import pytorch_lightning as pl

def create_ckpt_name_and_logger(ckpt_name):
    """Create a checkpoint name and logger for PyTorch Lightning.
    Args:
        ckpt_name (str): The name for the checkpoint.
    Returns:
        loggers (list): A list containing the TensorBoard and CSV loggers.
    """
    
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="../lightning_logs",  
        name="MultiMix_" + ckpt_name,  
    )

    csv_logger = pl.loggers.CSVLogger(
        save_dir="../lightning_logs",
        name="MultiMix_" + ckpt_name,
        version=tb_logger.version,  # Use the same version directory as the TensorBoard logger
    )
    loggers = [tb_logger, csv_logger]
    return loggers

