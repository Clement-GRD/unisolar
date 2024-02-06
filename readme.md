# Unisolar Project
This project is based on the **UNISOLAR** open dataset containing information on solar energy production and weather in various locations of the La Trobe University, Victoria, Australia, stored in a time series format.

The main goal of this project will be to forecast energy production based on previous production and weather data with a horizon of 24 hours.

## Structure
- `unisolar-project.ipynb` is the main notebook for the project.
- `unisolar_utils.py` contains the main functions used in the notebooks.
- `preprocessed_data.csv` contains a subset of preprocessed data (from `unisolar-project.ipynb`) for modeling.

## References
 - [Link](https://www.kaggle.com/datasets/cdaclab/unisolar/data) to dataset.
 - [GitHub repo](https://github.com/CDAC-lab/UNISOLAR/tree/main) containing subset of the data (not used here).
 - [Original paper](https://ieeexplore.ieee.org/document/9869474) on the UNISOLAR dataset (
    DOI: [10.1109/HSI55341.2022.9869474](https://ieeexplore.ieee.org/document/9869474)).

## Version History:
 - 2024-02-01: First commit
 - 2024-02-06: Added Weather EDA and utils file