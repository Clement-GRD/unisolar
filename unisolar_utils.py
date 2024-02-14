import pandas as pd 
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error

#############################################################################################
######################################    EDA    #############################################
#############################################################################################

def plot_solar_generation_sites(df, sites, figure_height=10, title=None):
    """
    Plots the evolution of `SolarGeneration`from the Dataframe `df` as a function of 'Timestamp' for selected sites.

    Parameters:
    - df (DataFrame): The DataFrame containing the solar generation data.
    - sites (list): A list of site number for which the solar generation will be plotted.
    - figure_height (int, optional): Height of the figure.
    - title (str, optional): Title for the entire plot.

    Returns:
    None

    This function creates subplots for each specified site, displaying the solar generation over time.
    The x-axis of all subplots is shared, providing a synchronized view.
    Each subplot is colored differently using the 'viridis' colormap, and legends are added for each site.
    Y-axis tick labels and ticks are removed, and a common x-axis label is added at the bottom of the plot.

    Example:
    plot_solar_generation_sites(solar_data, ['1', '2'], figure_height=8, title='Solar Generation Over Time')
    """
    cmap = mpl.colormaps['viridis'].resampled(len(sites))
    fig, axs = plt.subplots(len(sites), figsize=(15, figure_height), sharex=True, squeeze=False, gridspec_kw={'hspace': 0})
    for i, site in enumerate(sites):
        axs[i,0].plot(df.index, df[site], c=cmap(i), label=f'Site {site}')
        axs[i,0].legend(frameon=True, loc='center left')
        # axs[i,0].set_ylim(0)        
        axs[i,0].set(yticklabels=[])        # remove the ticks labels
        axs[i,0].tick_params(left=False)    # remove the ticks
    axs[-1,0].set_xlabel('Date')
    fig.suptitle(title, va='bottom')
    fig.text(0.1,0.5, "Solar Generation", ha="center", va="center", rotation=90)

def plot_weather_feature_campuses(df, column, campus, title=None):
    """
    Plots the specified weather feature from the DataFrame `df` for different campuses as a function of timestamp and print non-null data counts.

    Parameters:
    - df (DataFrame): The DataFrame containing the weather data.
    - column (str): The name of the weather feature to be plotted.
    - campus (list): A list of campus names for which the data will be plotted.
    - title (str, optional): Title for the entire plot.

    Returns:
    None

    This function creates subplots for each specified campus, displaying the weather feature over time.
    The x-axis of all subplots is shared, providing a synchronized view.
    Each subplot is colored differently using the 'viridis' colormap, and non-null data counts are printed.

    Example:
    plot_weather_feature_campus(weather_data, 'Temperature', ['CampusA', 'CampusB'], title='Temperature Variation')
    """
    cmap = mpl.colormaps['viridis'].resampled(len(campus))
    fig, axs = plt.subplots(len(campus), 1, figsize=(15, 7), sharex=True, squeeze=False) 
    for i, camp in enumerate(campus):
        axs[i, 0].scatter(df.index, df.loc[:, (column, camp)], color=cmap(i), s=1)
        axs[i, 0].scatter([], [], color=cmap(i), s=10, label=f'Campus {camp}')
        print(f'Campus {camp}: {df.loc[:, (column, camp)].notnull().values.sum()} non-null data')
        axs[i, 0].legend(loc='center left')
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0)

def plot_wind_orientation_campuses(df, campus_list):
    """
    Plots encoded wind direction for the 5 campuses as a function of timestamp, in different graphs.

    Parameters:
    - df (DataFrame): The DataFrame containing wind direction data for multiple campuses.
    - campus (list): A list of campus names for which the data will be plotted.


    Returns:
    None

    This function creates subplots for each campus, displaying the distribution of encoded wind direction over time.
    The x-axis of all subplots is shared, providing a synchronized view.
    Encoded wind direction values are visualized using count plots.

    Example:
    plot_windorientation_allcampus(wind_data)
    """
    fig, axs = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
    for i, camp in enumerate(campus_list):
        sns.countplot(x=df.loc[:, ('WindDirection', camp)], ax=axs[i], label=f'Campus {camp}')
        axs[i].legend(loc='center left')  
    fig.subplots_adjust(hspace=0)

#############################################################################################
#####################################    MODEL    ############################################
#############################################################################################

def naive_predictions(df: pd.DataFrame) -> pd.Series:
    """
    Generate naive solar generation predictions by shifting the 'SolarGeneration' column 24 hours into the future.

    Parameters:
    - df (DataFrame): Input DataFrame containing time series data.

    Returns:
    Series: A Series containing naive solar generation predictions where each value corresponds to the solar generation
            24 hours into the future. The first 24 values are filled with zeros.

    Example:
    >>> naive_preds = naive_predictions(data_frame)
    """
    return df['SolarGeneration'].copy().shift(24, fill_value=0)

def plot_naive_predictions(df: pd.DataFrame) -> None:
    """
    Plot true solar generation values, naive predictions, and prediction errors.

    Parameters:
    - df (pd.DataFrame): DataFrame containing time series data with 'SolarGeneration' column.

    Returns:
    None
    """
    predictions = naive_predictions(df)
    true_values = df['SolarGeneration']

    fig, axs = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axs[0].plot(true_values.index, true_values, label='true_values')
    axs[0].plot(predictions.index, predictions, label='naive_predictions')
    axs[1].plot(predictions.index, true_values - predictions, label='prediction_error')

    axs[1].set_xlabel('Date')
    axs[0].set_ylabel('Solar Generation')
    axs[1].set_ylabel('Prediction Error')
    axs[0].set_ylim(0, 8)

    axs[0].legend(loc=('upper left'))
    axs[1].legend(loc=('upper left'))
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Naive Model Predictions')


def print_naive_predictions_metrics(df: pd.DataFrame) -> None:
    """
    Print metrics for naive solar generation predictions.

    Parameters:
    - df (pd.DataFrame): DataFrame containing time series data with 'SolarGeneration' column.

    Returns:
    None
    """
    predictions = naive_predictions(df)
    true_values = df['SolarGeneration']

    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)

    print(f'Mean Squared Error: {mse: .3f}')
    print(f'Mean Absolute Error: {mae: .3f}')

def train_valid_test_split_df(df: pd.DataFrame, train_timestamp: str, test_timestamp: str) -> tuple:
    """
    Splits the DataFrame into training, validation, and test sets based on timestamp values: training for timestamp < train_timestamp,
    validation for train_timestamp <= timestamp < test_timestamp and test for test_timestamp <= timestamp.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - train_timestamp (str): The timestamp value used to split data into the training set.
    - test_timestamp (str): The timestamp value used to split data into the test set.

    Returns:
    tuple: A tuple containing df_train, df_valid, df_test.

    - df_train (DataFrame): Training set.
    - df_valid (DataFrame): Features for the validation set.
    - df_test (DataFrame): Features for the test set.


    Example:
    df_train, df_valid, df_test = train_test_valid_split_df(data, '2020-01-01', '2021-03-01')
    """    
    df_train = df[df.index < train_timestamp].copy()
    
    df_valid = df[(df.index >= train_timestamp) & (df.index < test_timestamp)].copy()
    
    df_test = df[df.index >= test_timestamp].copy()
    
    return df_train, df_valid, df_test,    

def plot_train_valid_test(df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Plot solar generation for training, validation, and test datasets as a function of time.

    Parameters:
    - df_train (pd.DataFrame): DataFrame for the training dataset.
    - df_valid (pd.DataFrame): DataFrame for the validation dataset.
    - df_test (pd.DataFrame): DataFrame for the test dataset.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, figsize=(15, 5))
    ax.plot(df_train['SolarGeneration'], label='Train')
    ax.plot(df_valid['SolarGeneration'], label='Valid')
    ax.plot(df_test['SolarGeneration'], label='Test')
    ax.legend(loc='lower left')
    ax.set_ylabel('Solar Generation')
    ax.set_xlabel('Date')
    ax.set_ylim(0, 8)
    fig.suptitle("Train, validation and test sets")

'''
See "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by A. Géron chapter 15 
and its GitHub repo
https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb
''' 
def to_windows(dataset, length):
    """
    Converts a time series dataset into a windowed dataset with specified window length.

    Parameters:
    - dataset (tf.data.Dataset): Input time series dataset.
    - length (int): Length of the desired windows.

    Returns:
    - tf.data.Dataset: Windowed dataset with windows of the specified length.

    Example:
    ```python
    raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    input_dataset = tf.data.Dataset.from_tensor_slices(raw_data)
    windowed_dataset = to_windows(input_dataset, length=3)
    ```

    Note:
    The function uses the `window` method to create overlapping windows of the specified length.
    The `flat_map` method is then applied to flatten the nested windows into a single dataset.
    The resulting dataset contains consecutive windows of the specified length.

    The `drop_remainder=True` parameter ensures that only complete windows are included in the dataset.
    """
    dataset = dataset.window(length, shift=1, drop_remainder=True) 
    return dataset.flat_map(lambda window_ds: window_ds.batch(length))

'''
See "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by A. Géron chapter 15 
and its GitHub repo
https://github.com/ageron/handson-ml3/blob/main/15_processing_sequences_using_rnns_and_cnns.ipynb
''' 
def to_seq2seq_dataset(series: np.ndarray,
                        seq_length: int = 48,
                        ahead: int = 24,
                        target_col: int = 1,
                        batch_size: int = 32,
                        shuffle: bool = False,
                        seed: int = None) -> tf.data.Dataset:   
    """
    Converts a time series dataset into a sequence-to-sequence dataset for training neural networks.

    Parameters:
    - series (np.ndarray or pd.Series): Time series data to be converted.
    - seq_length (int): Length of the input sequence.
    - ahead (int): Number of steps ahead for the target sequence.
    - target_col (int): Index of the target column in the original series.
    - batch_size (int): Size of each batch in the resulting dataset.
    - shuffle (bool): Whether to shuffle the dataset.
    - seed (int): Seed for reproducibility if shuffle is True.

    Returns:
    - tf.data.Dataset: A sequence-to-sequence dataset suitable for training neural networks.

    Example:
    ```python
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataset = to_seq2seq_dataset(series, seq_length=3, ahead=2, target_col=0, batch_size=2, shuffle=True, seed=42)
    ```

    Note:
    The resulting dataset will contain input sequences of length `seq_length` and target sequences
    of length `ahead`. The target sequence is extracted from the `target_col` column in the original series.

    If shuffle is True, the dataset will be shuffled with a specified seed for reproducibility.
    """
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1) 
    ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, target_col])) 
    if shuffle:
        ds = ds.shuffle(256 * batch_size, seed=seed) 
    return ds.batch(batch_size)

def plot_random_predictions(df: pd.DataFrame,
                          dict_of_predictions: dict[str, np.ndarray],
                          training_window: int,
                          prediction_window: int) -> None:
    """
    Plot selected predictions for solar generation along with labels and naive predictions.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - dict_of_predictions (Dict[str, np.ndarray]): Dictionary containing different prediction methods.
    - training_window (int): Number of data points used in the training set.
    - prediction_window (int): Number of steps ahead for prediction.

    Returns:
    None
 
    Example:
    ```python
    plot_random_predictions(data_frame, {'Method1': predictions1, 'Method2': predictions2}, 100, 24)
    ```

    Note:
    This function randomly selects a subset of data points and plots the corresponding solar generation labels,
    naive predictions, and predictions from different methods provided in the dictionary `dict_of_predictions`.
    """
    number_of_plots = 6

    fig, axs = plt.subplots(1, number_of_plots, figsize=(15, 5), sharey=True)

    if dict_of_predictions == {}:
        min_index_number = len(df)
    else:
        min_index_number = min([len(dict_of_predictions[x]) for x in dict_of_predictions])
    for a in range(number_of_plots):
        i = random.randrange(min_index_number)

        index = df.iloc[(training_window + i):training_window + i + prediction_window].index

        axs[a].plot(index, df['SolarGeneration'].iloc[(training_window + i):training_window + i + prediction_window].values, label='Label')
        axs[a].plot(index, df['SolarGeneration'].shift(-24).iloc[(training_window + i):training_window + i + prediction_window].values, ls=':', label='Naive pred.')

        for prediction in dict_of_predictions:
            axs[a].plot(index, dict_of_predictions[prediction][i, -1, :], ls='--', label=prediction)

        date_form = DateFormatter("%-I")  # hour without zero padding
        axs[a].xaxis.set_major_formatter(date_form)

    axs[0].legend(loc='upper center', frameon=False)
    axs[0].set_ylabel('Solar Generation')
    axs[0].set_ylim(0, 8)

    fig.subplots_adjust(wspace=0)
    fig.text(x=0.5, y=0, s='Hour')
    fig.suptitle('Random Solar Generation Predictions')

def plot_training_history(history, title_string: str) -> None:
    """
    Plot training and validation metrics from the history of a neural network.

    Parameters:
    - history (Any): History object returned by the `fit` method of a Keras model.
    - title_string (str): Title string for the plot.

    Returns:
    None

    Example:
    ```python
    plot_training_history(model_history, 'Training History')
    ```

    Note:
    This function plots training and validation metrics, including mean squared error (mse) and mean absolute error (mae),
    from the history object obtained during the training of a neural network.
    """
    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    fig.subplots_adjust(hspace=0)
    
    axs[0].plot(history.history['mse'], label='training')
    axs[0].plot(history.history['val_mse'], label='validation')
    axs[1].plot(history.history['mae'])
    axs[1].plot(history.history['val_mae'])
        
    axs[0].set_ylabel('mse')
    axs[0].set_yscale('log')
    axs[0].grid(which='both', axis='y', linestyle='--')

    axs[1].set_ylabel('mae')
    axs[1].set_yscale('log')
    axs[1].grid(which='both', axis='y', linestyle='--')

    axs[1].set_xlabel('Epochs')
    axs[0].set_title(title_string)

    axs[0].legend()
    

def prediction_df(predictions: np.array, df: pd.DataFrame, prediction_window: int, training_window: int) -> pd.DataFrame:
    """
    Create a DataFrame containing solar generation predictions and corresponding timestamps.

    Parameters:
    - predictions (Any): Predictions obtained from a prediction model.
    - df (pd.DataFrame): Original DataFrame containing time series data with 'SolarGeneration' target.
    - prediction_window (int): Number of steps ahead for the predictions.
    - training_window (int): Number of data points used in the training set.

    Returns:
    pd.DataFrame: A DataFrame with columns representing solar generation targets along with predictions for each step ahead.

    Example:
    ```python
    predictions = model.predict(input_data)
    prediction_dataframe = prediction_df(predictions, original_dataframe, 24)
    ```

    Note:
    This function takes predictions from a model, extracts the solar generation values,
    and organizes them into a DataFrame with columns representing each step ahead in the prediction horizon.
    """
    prediction_df = df['SolarGeneration'].iloc[training_window :-1].copy()
    for ahead in range(prediction_window):
        prediction_series = pd.Series(predictions[:-1, -1, ahead], df.iloc[training_window + ahead : -prediction_window + ahead].index,
                                    name=f'pred_{ahead}')
        prediction_df = pd.concat([prediction_df, prediction_series], axis=1) 
    return prediction_df

def create_error_list(prediction_df_model: pd.DataFrame, prediction_window: int) -> tuple[list[float], list[float]]:
    """
    Create lists of Mean Absolute Error (MAE) and Mean Squared Error (MSE) for each step in the prediction window.

    Parameters:
    - prediction_df_model (pd.DataFrame): DataFrame containing solar generation predictions and actual values.
    - prediction_window (int): Number of steps ahead in the prediction window.

    Returns:
    Tuple[List[float], List[float]]: Lists of MAE and MSE values for each step ahead.

    Example:
    ```python
    prediction_errors = create_error_list(prediction_dataframe, 24)
    ```

    Note:
    This function calculates the MAE and MSE between the actual solar generation values and predictions
    for each step ahead in the specified prediction window.
    """
    mae_list = []
    mse_list = []

    for ahead in range(prediction_window):
        column = f'pred_{ahead}'
        prediction = prediction_df_model[column].dropna()
        target = prediction_df_model.iloc[:, 0][prediction_df_model[f'pred_{ahead}'].notna()]
        mae = mean_absolute_error(target, prediction)
        mse = mean_squared_error(target, prediction)
        mae_list.append(mae)
        mse_list.append(mse)

    return mae_list, mse_list

def plot_errors_horizon(error_dict: dict[str, tuple[list[float], list[float]]]) -> None:
    """
    Plot Mean Absolute Error (MAE) and Mean Squared Error (MSE) for different prediction models.

    Parameters:
    - error_dict (Dict[str, Tuple[List[float], List[float]]]): Dictionary containing error lists for each model.

    Returns:
    None

    Example:
    ```python
    error_dict = {'Model1': ([mae1_1, mae1_2], [mse1_1, mse1_2]),
                  'Model2': ([mae2_1, mae2_2], [mse2_1, mse2_2])}
    plot_errors_horizon(error_dict)
    ```

    Note:
    This function takes a dictionary with keys as model names and values as tuples of MAE and MSE lists.
    It plots the errors for each model across different prediction horizons.

    The input dictionary should have the following structure:
    ```
    {'ModelName1': ([MAE_values], [MSE_values]),
     'ModelName2': ([MAE_values], [MSE_values]),
     ...}
    ```
    """
    fig, axs = plt.subplots(2, 1, sharex=True)
    
    for model, (mae_list, mse_list) in error_dict.items():
        axs[0].plot(mae_list, label=model)
        axs[1].plot(mse_list)

    axs[0].set_ylabel('MAE')
    axs[1].set_ylabel('MSE')

    axs[0].grid(which='both', axis='both', linestyle='--')

    axs[1].set_xlabel('Prediction Horizon (h)')
    axs[1].grid(which='both', axis='both', linestyle='--')

    axs[1].set_xlim(0, 24)
    axs[1].set_xticks(range(0, 25, 6))
    axs[0].legend()

    fig.subplots_adjust(hspace=0)
    fig.suptitle("Average Error as a Function of Prediction Horizon")

def main():
    return

if __name__ == '__main__':
    main()