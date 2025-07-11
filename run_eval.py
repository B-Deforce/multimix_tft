import argparse
import os

import pandas as pd
import pytorch_lightning as pl

from evaluation.evaluation import TrainedModel
from dataloader.custom_dataset import TimeSeriesDataLoader
from utils import get_config, read_json


def main(args):
    pl.seed_everything(96)
    print("Loading config: ", args.config_path)
    config = get_config(args.config_path)

    m_config = config["model_params"]

    # load data
    train_df = pd.read_parquet(config["train_path"])
    val_df = pd.read_parquet(config["val_path"])
    test_df = pd.read_parquet(config["test_path"])

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

    test_dataloader = data_module.test_dataloader()

    # load model
    model = TrainedModel(
        checkpoint_path=args.ckpt_path,
        task=args.task,
    )
    scaling_params = read_json(config["scaling_path"])

    # get predictions
    y, y_hat, x_all = model.predict(
        dataloader=test_dataloader,
        scaling_params=scaling_params,
        target_col=m_config["mf_target_col"],
        return_x=args.return_x,
    )

    # get metrics
    if "classification" not in args.task:
        mae, corr, mse, rmse = model.get_regression_metrics(
            y_true=y,
            y_pred=y_hat[:, len(model.model.quantiles) // 2]
            if model.model.quantiles
            else y_hat,
        )
        print("Metrics on test set:")
        print(f"MAE: {mae}")
        print(f"Correlation: {corr}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
    else:
        raise NotImplementedError(
            "Metrics for classification tasks are not implemented yet. "
            "These should be added in the `TrainedModel` class."
        )

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

        df = {"y": y}
        if model.model.quantiles is not None:
            for i, q in enumerate(model.model.quantiles):
                df[f"y_hat_{q}"] = y_hat[:, i]
        else:
            df["y_hat"] = y_hat
        df = pd.DataFrame(df)
        if args.return_x:
            for key, value in x_all.items():
                df[key] = value

        results_path = args.results_dir + "preds_test.parquet"
        df.to_parquet(results_path, index=False)
        print(f"Predictions saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MultiMix.")
    parser.add_argument("--config_path", type=str)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path with trained model.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to save predictions",
    )
    parser.add_argument(
        "--task",
        choices=[
            "regression",  # Standard regression task
            "classification",  # Classification task
            "regression_transfer_partial_ft",  # Transfer learning with partial fine-tuning
            "regression_transfer_full_ft",  # Transfer learning with full fine-tuning
        ],
        default="regression",
        help="Task type.",
    )
    parser.add_argument(
        "--return_x",
        type=bool,
        default=False,
        help="Return key input values: treatment indicator, field id, and crop-type",
    )
    args = parser.parse_args()
    main(args)
