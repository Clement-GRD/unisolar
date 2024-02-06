import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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