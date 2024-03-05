# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:44:18 2024

@author: syona
"""

import os
import pandas as pd








# Define the path to your file, including the extension if it has one
file_path = '/content/drive/MyDrive/Conservatives-Official/Water_Quality/uv'  # Make sure the extension is correct

# Define the column names based on the header of your data
columns = ['agency_cd', 'site_no', 'datetime', 'tz_cd', '92327_00060', '92327_00060_cd']

# Read the data, skipping the appropriate number of rows to reach the header
# Adjust the 'skiprows' if the headers are on a different line
df1 = pd.read_csv(file_path, sep='\t', skiprows=169, names=columns, on_bad_lines='skip')
print(df1.head())



df1.drop(['agency_cd','92327_00060_cd','tz_cd'], axis=1, inplace=True)

print(len(df1))

df1['92327_00060'] = df1['92327_00060'].replace('Ice', 0)

missing_values = df1['92327_00060'].isna()


missing_count = missing_values.sum()
print(missing_count)

df1.dropna(subset=['92327_00060'], inplace=True)
print(len(df1))




df1['datetime'] = pd.to_datetime(df1['datetime'], errors='coerce')
print(df1['datetime'].dtype)




#Extract date from datetime
df1['date'] = df1['datetime'].dt.date

print(df1.head(20))


df1['92327_00060'] = pd.to_numeric(df1['92327_00060'], errors='coerce')

# Remove rows where '74170_00060' is NaN, which are the rows with originally non-numeric values
df1 = df1.dropna(subset=['92327_00060'])



# Group by 'date', 'site_no', and 'tz_cd', then calculate the mean of '74170_00060'

aggregated_data1 = df1.groupby(['date', 'site_no'])['92327_00060'].mean().reset_index()

aggregated_data1['date_site'] = aggregated_data1['date'].apply(str) + "_" + aggregated_data1['site_no'].astype(str)

aggregated_data1["site_no"]= aggregated_data1["site_no"].astype(str)

print(aggregated_data1.head(12))


print(aggregated_data1.head(20))

df2 = pd.read_csv("/content/drive/MyDrive/Conservatives-Official/Water_Quality/county_mapping.csv")

df2["site_number"] = df2["site_number"].astype(str)


# Add leading zeros using list comprehension
df2["site_number"] = [f'0{site_number}' if len(str(site_number)) < 11 else site_number for site_number in df2["site_number"]]

print(df2.tail(60))



# Inner join on 'site_no' and 'site_number' columns
joined_df = aggregated_data1.merge(df2, left_on='site_no', right_on='site_number', how='inner')

joined_df = joined_df.drop(['site_number', 'location_dep', 'date_site'], axis=1)
joined_df = joined_df.rename(columns={'92327_00060': 'Streamflow'})
joined_df['date'] = pd.to_datetime(joined_df['date']).dt.strftime("%Y-%m-%d")

print(joined_df.head(10))


joined_df.to_csv('/content/drive/MyDrive/Conservatives-Official/Water_Quality/water_merged.csv', index=False)


