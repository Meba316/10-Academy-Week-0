# Import required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Define the folder containing your CSV files
data_folder = "data"  # Make sure this path is correct

# List all CSV files in the folder
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
df_list = []

# Loop through each file and read it
for file in csv_files:
    file_path = os.path.join(data_folder, file)
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Append the DataFrame to the list
        if not df.empty:
            df_list.append(df)
            print(f"Successfully loaded {file}")
        else:
            print(f"{file} is empty.")
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Check if any DataFrames were loaded
if df_list:
    # Concatenate all DataFrames into one DataFrame
    data = pd.concat(df_list, ignore_index=True)
    print(f"Data loaded successfully, shape of combined data: {data.shape}")
else:
    print("No valid CSV files were loaded.")
    
# Show the first few rows of the combined data
print(data.head())

# ------------------ Start Analysis ------------------

# 1. Calculate summary statistics
summary_stats = data.describe()
print("Summary Statistics:")
print(summary_stats)

# 2. Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# 3. Check for negative values in columns that should be positive (e.g., GHI, DNI, DHI)
negative_values = data[data[['GHI', 'DNI', 'DHI']] < 0]
print("\nNegative Values in GHI, DNI, DHI:")
print(negative_values)

# 4. Convert 'timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# 5. Plot GHI over time
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'], data['GHI'])
plt.title("Global Horizontal Irradiance (GHI) Over Time")
plt.xlabel("Time")
plt.ylabel("GHI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 6. Plot correlation matrix
correlation_matrix = data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 7. Wind speed distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['WS'], kde=True)
plt.title("Wind Speed Distribution")
plt.show()

# 8. Wind direction analysis
plt.figure(figsize=(8, 6))
sns.histplot(data['WD'], kde=True)
plt.title("Wind Direction Distribution")
plt.show()

# 9. Plot relationship between RH and temperature
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['RH'], y=data['Tamb'])
plt.title("Temperature vs. Relative Humidity")
plt.xlabel("Relative Humidity")
plt.ylabel("Temperature")
plt.show()


# 10. Plot histograms for GHI, DNI, DHI
plt.figure(figsize=(12, 6))
data[['GHI', 'DNI', 'DHI']].hist(bins=30)
plt.tight_layout()
plt.show()

# 11. Calculate Z-scores for outlier detection
z_scores = zscore(data[['GHI', 'DNI', 'DHI', 'WS']])
outliers = (z_scores > 3) | (z_scores < -3)
outliers_data = data[outliers.any(axis=1)]
print("\nOutliers based on Z-scores:")
print(outliers_data)

# 12. Bubble chart for GHI vs Tamb vs RH
plt.figure(figsize=(10, 6))
plt.scatter(data['GHI'], data['Tamb'], s=data['RH']*10, alpha=0.5)
plt.title("GHI vs Tamb vs RH")
plt.xlabel("GHI")
plt.ylabel("Tamb")
plt.show()

# Removing outliers based on Z-scores
data_no_outliers = data[(z_scores < 3) & (z_scores > -3)]
print(f"Data shape after removing outliers: {data_no_outliers.shape}")
