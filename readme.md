# MultiMix TFT and MultiMix LSTM Models

This repository contains the implementation of two multi-task learning models, `MultiMix_TFT` and `MultiMix_LSTM`, specifically designed to handle mixed-frequency (MF) data.

## MultiMix Temporal Fusion Transformer (TFT)

The `MultiMix_TFT` class is an implementation of the Temporal Fusion Transformer model extended to handle multi-task learning scenarios with mixed-frequency data. In other words, it is suitable for datasets where different features have different update frequencies. This model aims to leverage the temporal relationships between different frequency levels of data, thereby enhancing prediction capabilities. The TFT uses a combination of fixed and learnable parameters to capture both long and short-term temporal dynamics. 

### Features of the MultiMix TFT:
- Handles multi-task learning with mixed-frequency data.
- Utilizes the Temporal Fusion Transformer architecture.
- Supports multiple input feature types: continuous, categorical, and datetime.

## MultiMix LSTM

The `MultiMix_LSTM` class is a Long Short-Term Memory (LSTM) model extended for multi-task learning scenarios with mixed-frequency data. Similar to the TFT, it can handle datasets where different features have different update frequencies. The LSTM model is a type of recurrent neural network (RNN) capable of learning long-term dependencies.

### Features of the MultiMix LSTM:
- Handles multi-task learning with mixed-frequency data.
- Utilizes an LSTM architecture with multiple layers and dropout for regularization.

## Data Sources

The models `MultiMix_TFT` and `MultiMix_LSTM` have been designed and tested using proprietary mixed-frequency datasets. As these datasets are confidential and not open-source, they are not included in this repository. However, the models are flexible and can work with a wide range of mixed-frequency datasets. We encourage you to apply these models to your own mixed-frequency datasets, whether they are sourced from public domains or your proprietary data.

We are actively exploring the possibilities of sharing a synthetic mixed-frequency dataset that mirrors the characteristics of the proprietary datasets in a non-identifiable manner. However, we hope that the provided code will serve as a good starting point for those interested in exploring the capabilities of these models.

In the meantime, please refer to the original paper "MultiMix TFT: A Multi-task Mixed-Frequency Framework with Temporal Fusion Transformers" (Deforce, Baesens, Diels, & Serral Asensio, 2023) for a deeper understanding of the concept and the methodology employed, to appear soon in Proceedings of Machine Learning Research (PMLR) for the 2nd Conference on Lifelong Learning Agents ([CoLLAs](https://lifelong-ml.cc))
