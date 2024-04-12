# MultiMix Framework for Multi-Task Learning with Mixed-Frequency Data

This repository contains the implementation of `MultiMix` as described in [MultiMix TFT: A Multi-task Mixed-Frequency Framework with Temporal Fusion Transformers](https://proceedings.mlr.press/v232/deforce23a.html)

`MultiMix` is implemented with the Temporal Fusion Transformer [1], extended to handle multi-task learning scenarios with mixed-frequency data. In other words, it is suitable for datasets where different features have different update frequencies. This model aims to leverage the temporal relationships between different frequency levels of data, thereby enhancing prediction capabilities. The TFT uses a combination of fixed and learnable parameters to capture both long and short-term temporal dynamics. 

### Features of the MultiMix TFT:
- Handles multi-task learning with mixed-frequency data.
- Utilizes the Temporal Fusion Transformer [1] architecture.
- Supports multiple input feature types: continuous, categorical, and datetime.

[1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748-1764.

## Baselines

`Baselines` contains an implementation of `MultiMix` with Long Short-Term Memory (LSTM) model extended for multi-task learning scenarios with mixed-frequency data.

## Data Sources

`MultiMix` has been designed and tested using proprietary mixed-frequency datasets. As these datasets are confidential and not open-source, they are not included in this repository. However, the models are flexible and can work with a wide range of mixed-frequency datasets. We encourage you to apply these models to your own mixed-frequency datasets, whether they are sourced from public domains or your proprietary data.

We are actively exploring the possibilities of sharing a synthetic mixed-frequency dataset that mirrors the characteristics of the proprietary datasets in a non-identifiable manner. However, we hope that the provided code will serve as a good starting point for those interested in exploring the capabilities of `MultiMix`.

Please cite the following paper if you use this code in your research:

```bibtex
@inproceedings{deforce2023multimix,
  title={MultiMix TFT: A Multi-task Mixed-Frequency Framework with Temporal Fusion Transformers},
  author={Deforce, Boje and Baesens, Bart and Diels, Jan and Asensio, Estefan{\'\i}a Serral},
  booktitle={Conference on Lifelong Learning Agents},
  pages={586--600},
  year={2023},
  organization={PMLR}
}
```
