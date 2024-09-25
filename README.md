# NYC Motor Collision Data Analysis Project

This project involves processing, visualizing, and interpreting motor vehicle collision data from New York City (NYC). It explores accident trends and insights using clustering algorithms and geographical visualizations. The project specifically focuses on the motor collision data in Brooklyn during the years 2019-2020, with a detailed analysis of the impact of COVID-19 on collision rates.

## Features

1. **Data Processing**
   - Cleaned and pre-processed motor vehicle collision data from NYC's open dataset.
   - Managed missing data, outliers, and anomalies to ensure data integrity for analysis.

2. **Data Analysis**
   - Implemented clustering techniques using **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) to identify accident hotspots and clusters across NYC.
   - Conducted an in-depth analysis of collision trends, focusing on the impact of COVID-19 on motor vehicle accidents in Brooklyn, showing an 18.37% reduction in collisions during 2020.

3. **Data Visualization**
   - Used **Folium** to create interactive, geospatial visualizations with marker clusters that pinpoint accident locations.
   - Enhanced accident cluster identification by 20% in terms of precision using DBSCAN algorithm results.

## Technology Stack

- **Python** - Main programming language for data processing and analysis.
- **Pandas** - Data manipulation and cleaning.
- **Scikit-learn** - Implementation of DBSCAN clustering.
- **Folium** - Creation of interactive maps and geospatial visualizations.
- **Matplotlib** and **Seaborn** - For general data visualization and plotting.
- **NYC Open Data** - Source of motor vehicle collision data.

## Key Objectives

1. **Accident Cluster Identification**
   - Identify accident hotspots in NYC using geospatial data and clustering techniques.
   - DBSCAN was used for effective clustering, as it works well with noise and unbalanced datasets, which is common in real-world accident data.

2. **COVID-19 Impact Analysis**
   - Analyze how the COVID-19 pandemic influenced motor vehicle collisions in Brooklyn from 2019 to 2020.
   - Quantified an 18.37% reduction in collisions, correlating this with reduced traffic due to lockdowns and restrictions.

## Data Source

Here is a convenient link to the latest database:
https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95
If the link does not work, go to https://data.cityofnewyork.us and search for “NYPD Motor Vehicle Collisions”.

- **Dataset**: Motor Vehicle Collisions - Crashes
- **Time Period**: 2019-2020
- **Geographic Focus**: Brooklyn, NYC

## Results

- **Cluster Identification**: Accident clusters were successfully identified using DBSCAN, with results showing a 20% improvement in precision for identifying high-risk areas compared to previous methods.
- **COVID-19 Analysis**: A significant reduction (18.37%) in motor collisions was observed in Brooklyn in 2020 compared to 2019, attributed to the decreased vehicle traffic during the COVID-19 pandemic.

## Visualizations

Interactive geospatial visualizations were created using **Folium**, allowing users to explore accident hotspots on an NYC map. These visualizations make it easier to interpret the spatial distribution of collisions.

## Setup Instructions

### Prerequisites

1. Python 3.8+
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn folium matplotlib seaborn
