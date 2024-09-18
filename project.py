"""
Author 1 : Sahil Sanjay Gunjal. (sg2736)
Author 2 : Siddhesh Abhijeet Dhonde. (sd1386)
Python Version: 3.10
Project : Data Mining on Motor_vehicle_collisions in NYC dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import MarkerCluster


def divide_the_data(brooklyn_dataframe):
    """
    This function divides the data into summer of 2019 and 2020.
    :param brooklyn_dataframe:
    :return: Dataframes of summer 2019 and 2020
    """
    start_date_2019 = pd.to_datetime('06-01-2019')
    end_date_2019 = pd.to_datetime('07-31-2019')

    start_date_2020 = pd.to_datetime('06-01-2020')
    end_date_2020 = pd.to_datetime('07-31-2020')

    dataframe_2019 = brooklyn_dataframe[
        (brooklyn_dataframe['CRASH DATE'] >= start_date_2019) & (brooklyn_dataframe['CRASH DATE'] <= end_date_2019)]
    dataframe_2020 = brooklyn_dataframe[
        (brooklyn_dataframe['CRASH DATE'] >= start_date_2020) & (brooklyn_dataframe['CRASH DATE'] <= end_date_2020)]

    return dataframe_2019, dataframe_2020


def read_data():
    """
    This function reads the data and convert in dataframe
    :return: Dataframe of data.
    """
    dataframe = pd.read_csv('Motor_Vehicle_Collisions_-_Crashes_20231201.csv')
    return dataframe


def select_boroughs(dataframe):
    """
    This funtion selects BOROUGH as BROOKLYN and Makes another dataframe of it.
    :param dataframe: Dataframe of whole data
    :return: Dataframe of brooklyn data.
    """
    brooklyn_dataframe = dataframe[dataframe['BOROUGH'] == 'BROOKLYN']
    brooklyn_dataframe.reset_index(drop=True, inplace=True)
    brooklyn_dataframe.to_csv('brooklyn_collisions.csv', index=False)
    return brooklyn_dataframe


def plot_for_2019_2020_summer(dataframe):
    """
    This function plots graph for summer 2019 and 2020
    :param dataframe: Dataframe of brooklyn
    :return: None
    """
    start_date_2019 = '2019-05-01'
    end_date_2019 = '2019-08-01'

    start_date_2020 = '2020-05-01'
    end_date_2020 = '2020-08-01'
    # Create a DataFrame with a range of dates between start and end dates
    date_range_2019 = pd.DataFrame({'Date': pd.date_range(start=start_date_2019, end=end_date_2019)})
    date_range_2020 = pd.DataFrame({'Date': pd.date_range(start=start_date_2020, end=end_date_2020)})

    # Display the DataFrame
    date_range_2019_ls = date_range_2019['Date'].to_list()
    date_range_2020_ls = date_range_2020['Date'].to_list()

    summer_2019 = []
    summer_2020 = []

    for i in range(len(date_range_2020_ls) - 1):
        temp_1 = dataframe[(dataframe['CRASH DATE'] >= date_range_2019_ls[i]) & (
                    dataframe['CRASH DATE'] <= date_range_2019_ls[i + 1])].shape[0]
        temp_2 = dataframe[(dataframe['CRASH DATE'] == date_range_2020_ls[i]) & (
                    dataframe['CRASH DATE'] <= date_range_2020_ls[i + 1])].shape[0]

        summer_2019.append(temp_1)
        summer_2020.append(temp_2)

    plt.figure(figsize=(10, 6))
    plt.plot(date_range_2019_ls[:len(date_range_2019_ls) - 1], summer_2019, label='2019')
    plt.plot(date_range_2019_ls[:len(date_range_2019_ls) - 1], summer_2020, label='2020')

    plt.title(f'Number of Accidents Per Day in Summer of 2019 and 2020')
    plt.xlabel('Days of the Summer')
    plt.ylabel('Number of Accidents')

    # Format the x-axis to show only the month and day
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    plt.legend()
    plt.show()


def create_scatter_for_difference(dataframe_2019, dataframe_2020, start_date_2019, end_date_2019, start_date_2020,
                                  end_date_2020, month):
    """
    This function is used to plot the graph for accidents on each date of june and july of 2019 and 2020 separately.
    :param dataframe_2019: Data of 2019
    :param dataframe_2020: Data of 2020
    :param start_date_2019: Start date of 2019
    :param end_date_2019: end date of 2019
    :param start_date_2020: start date of 2020
    :param end_date_2020: end date of 2020
    :param month: June/July
    :return: None
    """
    dataframe_j_2019 = dataframe_2019[
        (dataframe_2019['CRASH DATE'] >= start_date_2019) & (dataframe_2019['CRASH DATE'] <= end_date_2019)]
    dataframe_j_2020 = dataframe_2020[
        (dataframe_2020['CRASH DATE'] >= start_date_2020) & (dataframe_2020['CRASH DATE'] <= end_date_2020)]

    accidents_j_2019 = dict()
    accidents_j_2020 = dict()

    for index, row in dataframe_j_2019.iterrows():
        crash_date = row['CRASH DATE']

        if crash_date.day not in accidents_j_2019:
            accidents_j_2019[crash_date.day] = 1

        else:
            accidents_j_2019[crash_date.day] += 1

    for index, row in dataframe_j_2020.iterrows():
        crash_date = row['CRASH DATE']
        if crash_date.day not in accidents_j_2020:
            accidents_j_2020[crash_date.day] = 1

        else:
            accidents_j_2020[crash_date.day] += 1

    list_j_2019 = []
    list_j_2020 = []
    if month == 'June':
        days = 30
    else:
        days = 31
    for i in range(days):
        if i in accidents_j_2019:
            list_j_2019.append(accidents_j_2019[i])
        else:
            list_j_2019.append(0)

        if i in accidents_j_2020:
            list_j_2020.append(accidents_j_2020[i])

        else:
            list_j_2020.append(0)

    days = np.arange(1, len(list_j_2019) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(days, list_j_2019, label=f'{month} 2019')
    plt.plot(days, list_j_2020, label=f'{month} 2020')

    plt.title(f'Number of Accidents Per Day in {month}')
    plt.xlabel('Day')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.show()


def analyze_100_consecutive_days(dataframe):
    """
    This function plots 100 consecutive days which had most number of accidents
    :param dataframe: Dataframe of brooklyn
    :return: None
    """
    # Sort the DataFrame based on the 'CRASH DATE' column
    dataframe = dataframe.sort_values(by='CRASH DATE')
    # dataframe.set_index('CRASH DATE', inplace=True)

    start_date = '2019-01-01'
    end_date = '2020-10-31'

    # Create a DataFrame with a range of dates between start and end dates
    date_range = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date)})

    # Display the DataFrame
    date_range_df = date_range['Date'].to_list()
    ptr = 0
    ptr2 = 99
    accidents = []
    each_day_accidents = []
    max_days = []
    idx1 = 0
    idx2 = 0
    while ptr2 < len(date_range_df):
        temp = dataframe[
            (dataframe['CRASH DATE'] >= date_range_df[ptr]) & (dataframe['CRASH DATE'] <= date_range_df[ptr2])].shape[0]
        # print(temp)

        accidents.append((temp, date_range_df[ptr], date_range_df[ptr2]))
        ptr += 1
        ptr2 += 1
    max_acc = 0
    ans = None

    for i in accidents:
        if i[0] > max_acc:
            max_acc = i[0]
            ans = i

    for i in range(len(date_range_df)):
        temp = dataframe[(dataframe['CRASH DATE'] == date_range_df[i])].shape[0]
        each_day_accidents.append(temp)
        if date_range_df[i] == ans[1]:
            idx1 = i

        if date_range_df[i] == ans[2]:
            idx2 = i

    plt.figure(figsize=(15, 6))
    plt.plot(date_range_df, each_day_accidents, linestyle='-', label='Number of accidents on all days',
             color='lightblue')
    plt.plot(date_range_df[idx1:idx2 + 1], each_day_accidents[idx1:idx2 + 1],
             label='Consecutive 100 days with most accident', linestyle='-', color='red')
    plt.title('Number of Accidents by days')
    plt.xlabel('Date ranges')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.grid(True)
    plt.show()


def accidents_by_day_of_week(dataframe):
    """
    This function plots number accidents by day of the week.
    :param dataframe: Dataframe of brooklyn
    :return: None
    """
    dataframe['Day of Week'] = dataframe['CRASH DATE'].dt.day_name()
    Days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    accidents = []
    for i in Days:
        temp = dataframe[
            (dataframe['Day of Week'] == i)
        ].shape[0]
        accidents.append(temp)

    plt.figure(figsize=(10, 6))
    plt.plot(Days, accidents, marker='o', linestyle='-', color='b', label='Number of accidents each day of week')

    plt.title('Number of Accidents by days of the week')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.grid(True)
    plt.show()


def accidents_by_hour(dataframe):
    """
    This function plots number of accidents by hour of the day.
    :param dataframe: Dataframe of brooklyn
    :return: None
    """
    hours_of_day = [f"{hour:02}:00" for hour in range(24)]
    dataframe['CRASH TIME'] = pd.to_datetime(dataframe['CRASH TIME'])
    # Create 'Hour of Day' column
    dataframe['Hour of Day'] = dataframe['CRASH TIME'].dt.strftime('%H:00')
    # Display the list
    accidents = []
    for i in range(len(hours_of_day)):
        temp = dataframe[(dataframe['Hour of Day'] == hours_of_day[i])].shape[0]
        accidents.append(temp)

    plt.figure(figsize=(10, 6))
    plt.plot(hours_of_day, accidents, marker='o', linestyle='-', color='b', label='Number of accident at hour')

    plt.title('Number of Accidents by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Accidents')
    plt.legend()
    plt.grid(True)
    plt.show()


def most_12_days_of_accident_2020(dataframe):
    """
    This function plots 12 days of most accidents in 2020.
    :param dataframe: Dataframe of brooklyn
    :return: None
    """
    dataframe = dataframe[(dataframe['CRASH DATE'] >= '01-01-2020') & (dataframe['CRASH DATE'] <= '12-31-2020')]
    grouped_data = dataframe.groupby('CRASH DATE').size().reset_index(name='Number of Accidents')
    sorted_accidents = grouped_data.sort_values(by='Number of Accidents', ascending=False)
    all_dates = grouped_data['CRASH DATE'].tolist()
    all_accident_num = grouped_data['Number of Accidents'].tolist()

    # Display the grouped data
    top_dates = sorted_accidents.head(12)['CRASH DATE'].tolist()
    top_num_of_crashes = sorted_accidents.head(12)['Number of Accidents'].tolist()

    for i in top_dates:
        print(i)

    plt.figure(figsize=(15, 6))
    plt.plot(all_dates, all_accident_num, linestyle='-', label='All Accident Days', color='lightblue')

    # Scatter plot for the top 12 accident days in red
    plt.scatter(top_dates, top_num_of_crashes, color='red', label='Top 12 Accident Days', zorder=5)

    plt.title('Number of Accidents by days')
    plt.xlabel('Date ranges')
    plt.ylabel('Number of Accidents')
    plt.grid(True)
    plt.legend()
    plt.show()


def using_DBScan(df):
    """
    This function performs DBScan clustering on Longitude and Latitude of Brooklyn Data.
    :param df: Brooklyn Dataframe
    :return: None
    """
    df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    # Extract latitude and longitude columns
    coordinates = df[['LATITUDE', 'LONGITUDE']]
    filtered_df = df[(df['LATITUDE'] >= 35) & (df['LONGITUDE'] <= 70)]

    # Extract latitude and longitude columns
    coordinates = filtered_df[['LATITUDE', 'LONGITUDE']]

    # Calculate average distance
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(coordinates)
    distances, _ = neighbors.kneighbors(coordinates)
    avg_distance = distances.mean()

    # Set eps as a fraction of the average distance
    eps = 0.1 * avg_distance
    min_samples = 10  # Adjust according to your data and requirements
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

    # Fit DBScan to the coordinates
    filtered_df['Cluster'] = dbscan.fit_predict(coordinates)

    # Plot only cluster points (exclude noise points)
    clustered_df = filtered_df[filtered_df['Cluster'] != -1]

    # Get unique clusters and their sizes
    unique_clusters, cluster_sizes = np.unique(clustered_df['Cluster'], return_counts=True)

    # Create a colormap based on the number of nodes in each cluster
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(cluster_sizes.min(), cluster_sizes.max())
    colors = cmap(norm(cluster_sizes))

    plt.figure(figsize=(10, 8))
    plt.scatter(
        clustered_df['LONGITUDE'],
        clustered_df['LATITUDE'],
        c=clustered_df['Cluster'],
        cmap='viridis',
        s=10,
        edgecolors='face',
        alpha=0.7,
        marker='o',
        linewidths=0.5,
        facecolors=colors
    )

    plt.title('DBScan Clustering of Accidents')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def all_collions_on_map(brooklyn_dataframe):
    """
    This function uses marker cluster to plot locations on Map.
    :param brooklyn_dataframe: brooklyn Dataframe
    :return:
    """
    brooklyn_map = folium.Map(location=[40.6782, -73.9442], zoom_start=12)

    marker_cluster = MarkerCluster().add_to(brooklyn_map)

    for index, row in brooklyn_dataframe.iterrows():
        if not pd.isnull(row['LATITUDE']) and not pd.isnull(row['LONGITUDE']):

            popup_text = f"Crash Date: {row['CRASH DATE']}<br>Crash Time: {row['CRASH TIME']}<br>Location: {row['LOCATION']}"

            folium.Marker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                popup=popup_text,
                icon=None
            ).add_to(marker_cluster)

    brooklyn_map.save("brooklyn_accidents_map.html")

def main():
    """
    This is main function which drives the code
    :return: None
    """
    dataframe = read_data()
    brooklyn_dataframe = select_boroughs(dataframe)
    brooklyn_dataframe = pd.read_csv('brooklyn_collisions.csv')

    brooklyn_dataframe['CRASH DATE'] = pd.to_datetime(brooklyn_dataframe['CRASH DATE'])
    plot_for_2019_2020_summer(brooklyn_dataframe)
    dataframe_2019, dataframe_2020 = divide_the_data(brooklyn_dataframe)

    start_date_2019 = pd.to_datetime('06-01-2019')
    end_date_2019 = pd.to_datetime('06-30-2019')

    start_date_2020 = pd.to_datetime('06-01-2020')
    end_date_2020 = pd.to_datetime('06-30-2020')
    create_scatter_for_difference(dataframe_2019, dataframe_2020, start_date_2019, end_date_2019,
                                  start_date_2020, end_date_2020, 'June')

    start_date_2019 = pd.to_datetime('07-01-2019')
    end_date_2019 = pd.to_datetime('07-31-2019')

    start_date_2020 = pd.to_datetime('07-01-2020')
    end_date_2020 = pd.to_datetime('07-31-2020')

    create_scatter_for_difference(dataframe_2019, dataframe_2020, start_date_2019, end_date_2019,
                                  start_date_2020, end_date_2020, 'July')

    start_date_2019 = pd.to_datetime('01-01-2019')
    end_date_2020 = pd.to_datetime('10-31-2020')
    #
    dataframe_2019_2020 = brooklyn_dataframe[
        (brooklyn_dataframe['CRASH DATE'] >= start_date_2019) & (brooklyn_dataframe['CRASH DATE'] <= end_date_2020)]

    analyze_100_consecutive_days(dataframe_2019_2020)
    accidents_by_day_of_week(brooklyn_dataframe)
    accidents_by_hour(brooklyn_dataframe)
    most_12_days_of_accident_2020(brooklyn_dataframe)

    using_DBScan(brooklyn_dataframe)
    all_collions_on_map(brooklyn_dataframe)


# Start of the code
if __name__ == '__main__':
    main()
