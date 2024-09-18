import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

def analysis_by_vehicle_type(dataframe):
    """
    This function analyzes the brooklyn data and shows top 10 vehicle types which are causing accidents.
    :param dataframe: Brooklyn Dataframe
    :return: None
    """
    # Count the number of accidents for each vehicle type
    df = pd.DataFrame(dataframe)

    # Select relevant columns
    vehicle_columns = [col for col in df.columns if 'VEHICLE TYPE CODE' in col]

    # Melt the DataFrame to convert columns to rows
    melted_df = pd.melt(df, id_vars=['CRASH DATE'], value_vars=vehicle_columns, value_name='Vehicle Type')

    # Drop rows with empty, 'Unspecified', 'UNKNOWN', 'OTHER' 'Vehicle Type' values
    melted_df = melted_df.dropna(subset=['Vehicle Type'])
    melted_df = melted_df[
        (melted_df['Vehicle Type'] != 'Unspecified') &
        (melted_df['Vehicle Type'] != 'UNKNOWN') &
        (melted_df['Vehicle Type'] != 'OTHER')
        ]

    # Count the occurrences of each vehicle type
    vehicle_counts = melted_df['Vehicle Type'].value_counts()

    plt.figure(figsize=(12, 6))
    # Use plt.bar instead of plt.plot for a bar chart
    plt.bar(vehicle_counts.head(10).index, vehicle_counts.head(10), color='lightblue')
    plt.title('Number of Accidents by Vehicle Type')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=15, ha='right')
    plt.show()


def number_of_streets_danger_for_persons(dataframe):
    """
    This function analyzes the Brooklyn data gives top 10 streets which had most persons killed or injured.
    :param dataframe: Brooklyn Dataframe
    :return: None
    """
    street_names = dataframe['ON STREET NAME'].unique().tolist()

    # Create a dictionary to store the total number of killed and injured persons for each street
    street_statistics = {}

    # Iterate through each street
    for street in street_names:
        temp = dataframe[(dataframe['ON STREET NAME'] == street) & (
                (dataframe['NUMBER OF PERSONS KILLED'] > 0) | (dataframe['NUMBER OF PERSONS INJURED'] > 0))]
        total_fatalities = temp['NUMBER OF PERSONS KILLED'].sum()
        total_injuries = temp['NUMBER OF PERSONS INJURED'].sum()
        street_statistics[street] = total_fatalities + total_injuries

    # Sort streets based on the total number of killed and injured persons in descending order
    sorted_streets = sorted(street_statistics.items(), key=lambda x: x[1], reverse=True)

    # Take the top 10 streets
    top_streets = dict(sorted_streets[:10])
    streets_danger = []
    for i in top_streets.keys():
        streets_danger.append(i.strip())
    # Plot the top 10 streets
    plt.figure(figsize=(10, 6))
    plt.bar(streets_danger, top_streets.values(), color='blue')
    plt.title('Top 10 Streets with Most Killed and Injured Persons')
    plt.xlabel('Street Name')
    plt.ylabel('Total Killed and Injured Persons')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()


def streets_with_most_accidents(dataframe):
    """
    This function analyzes Brooklyn Dataframe and gives top 10 streets with most accidents
    :param dataframe: Brooklyn Dataframe
    :return: None
    """
    street_names = dataframe['ON STREET NAME'].unique().tolist()

    # Create a dictionary to store the total number of killed and injured persons for each street
    streets_accidents = []
    for street in street_names:
        temp = dataframe[dataframe['ON STREET NAME'] == street]
        streets_accidents.append((temp.shape[0], street))

    streets_accidents = sorted(streets_accidents, key=lambda x: x[0], reverse=True)
    streets = []
    num_of_accidents = []
    for i in streets_accidents[:11]:
        streets.append(i[1].strip())
        num_of_accidents.append(i[0])

    plt.figure(figsize=(10, 6))
    plt.bar(streets, num_of_accidents, color='blue')
    plt.title('Top 10 Streets with Most Accidents in Brooklyn')
    plt.xlabel('Street Name')
    plt.ylabel('Total Number of Accidents')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()


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


def top_accident_streets_pedestrians(brooklyn_dataframe):
    """
    This function gives streets which had most accidents related to pedestrians
    :param brooklyn_dataframe: Brooklyn Dataframe
    :return: None
    """
    # Extract relevant columns
    pedestrian_columns = ['ON STREET NAME', 'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED']

    # Create a new DataFrame with relevant columns
    pedestrian_data = brooklyn_dataframe[pedestrian_columns]

    # Group by 'ON STREET NAME' and sum the counts for pedestrians injured or killed
    street_pedestrian_count = pedestrian_data.groupby('ON STREET NAME').sum()

    # Filter out streets where both counts for pedestrians killed and pedestrians injured are zero
    street_pedestrian_count = street_pedestrian_count[(street_pedestrian_count['NUMBER OF PEDESTRIANS INJURED'] > 0) | (
            street_pedestrian_count['NUMBER OF PEDESTRIANS KILLED'] > 0)]

    # Calculate the total count of pedestrians injured or killed
    street_pedestrian_count['TOTAL_PEDESTRIANS'] = street_pedestrian_count['NUMBER OF PEDESTRIANS INJURED'] + \
                                                   street_pedestrian_count['NUMBER OF PEDESTRIANS KILLED']

    # Extract only the 'ON STREET NAME' and 'TOTAL_PEDESTRIANS' columns
    result_df = street_pedestrian_count[['TOTAL_PEDESTRIANS']].reset_index()

    # Sort the DataFrame by the total count in descending order
    result_df = result_df.sort_values(by='TOTAL_PEDESTRIANS', ascending=False).head(10)

    street_names = []
    for index, row in result_df.iterrows():
        temp_name = (row['ON STREET NAME']).strip()
        street_names.append(temp_name)

    plt.bar(street_names, result_df['TOTAL_PEDESTRIANS'])

    plt.xlabel('Street Name')
    plt.ylabel('Total Pedestrians (Injured or Killed)')
    plt.title('Top 10 Streets by Total Pedestrians Involved')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Show the plot
    plt.show()



def count_vehicle(brooklyn_dataframe):
    """
    This function plots graph showing number of vehicles involves in accidents.
    :param brooklyn_dataframe: brooklyn dataframe
    :return: none
    """
    # Extract relevant columns for vehicle types
    vehicle_columns = ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4',
                       'VEHICLE TYPE CODE 5']

    # Create a new column 'TOTAL_VEHICLES' representing the total number of non-empty vehicle entries in each row
    brooklyn_dataframe['TOTAL_VEHICLES'] = brooklyn_dataframe[vehicle_columns].apply(
        lambda row: row[row.notnull()].count(), axis=1)

    # Count occurrences of each total vehicle count
    vehicle_count_distribution = brooklyn_dataframe['TOTAL_VEHICLES'].value_counts().sort_index()

    # Print or display the results
    total_vehicle_and_accident = list(vehicle_count_distribution)

    x_values = np.arange(6)
    plt.bar(x_values, total_vehicle_and_accident)
    plt.xlabel('Number of Vehicles Involved')
    plt.ylabel('Number of Accidents')
    plt.title('Distribution of Accidents by Number of Vehicles Involved')
    plt.show()


def plot_heatmap(brooklyn_dataframe):
    """
    This function plots heatmap of accidents in brooklyn.
    :param brooklyn_dataframe: Brooklyn Dtaframe
    :return: None
    """
    # Create a base map centered around Brooklyn, NYC
    brooklyn_map_heatmap = folium.Map(location=[40.6782, -73.9442], zoom_start=12)

    # Ensure your data is loaded into a DataFrame (replace the sample data with your actual dataset)

    # Drop rows with missing latitude or longitude values
    df = brooklyn_dataframe.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # Convert the DataFrame to a list of lists (required format for HeatMap)
    heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in df.iterrows()]

    # Create a HeatMap layer
    HeatMap(heat_data).add_to(brooklyn_map_heatmap)

    # Save the map to an HTML file or display it
    brooklyn_map_heatmap.save("brooklyn_accidents_heatmap.html")

def main():
    """
    This is main function which drives the code.
    :return:
    """
    dataframe = read_data()
    brooklyn_dataframe = select_boroughs(dataframe)
    analysis_by_vehicle_type(brooklyn_dataframe)
    number_of_streets_danger_for_persons(brooklyn_dataframe)
    streets_with_most_accidents(brooklyn_dataframe)
    top_accident_streets_pedestrians(brooklyn_dataframe)
    count_vehicle(brooklyn_dataframe)
    plot_heatmap(brooklyn_dataframe)


# Start of the code
if __name__ == '__main__':
    main()
