# Unisolar Project
This project is based on the **UNISOLAR** open dataset containing information on solar energy production and weather in various locations of the La Trobe University, Victoria, Australia, stored in a time series format.

The goal of this project is to build a DL model using RNN and CNN to forecast the energy production for the next 24 hours based on solar energy production and weather data of the last 7 days. Our model performance will be evaluated against a naive model predicting the value recorded 24 hours in the past.

## Structure
- `unisolar_eda.ipynb` is the main EDA notebook for the project.
- `unisolar_model.ipynb` is the main modeling notebook for the project.
- `unisolar_utils.py` contains the main functions used in the notebooks.
- `preprocessed_data.csv` contains a subset of preprocessed data (from `unisolar-project.ipynb`) for modeling.

## References
 - [Link](https://www.kaggle.com/datasets/cdaclab/unisolar/data) to dataset.
 - [GitHub repo](https://github.com/CDAC-lab/UNISOLAR/tree/main) containing subset of the data (not used here).
 - [Original paper](https://ieeexplore.ieee.org/document/9869474) on the UNISOLAR dataset (
    DOI: [10.1109/HSI55341.2022.9869474](https://ieeexplore.ieee.org/document/9869474)).

## Version History:
 - 2024-02-01: First commit
 - 2024-02-06: Added Weather EDA, utils file and modeling notebook.
 - 2024-02-10: Added RNN and CNN models