import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
CSV_FILE = 'combined_data.csv'  # Input file

# Load the data from the CSV file
df = pd.read_csv(CSV_FILE)
# Filter out rows where Action is "none"
df = df[df['Action'] != 'none']
# Convert 'Date' to datetime for better plotting
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Fill missing values for DipCount (if any) with 0 for visualization purposes
df['DipCount'] = df['DipCount'].fillna(0)
# Visualization 1: Boxplot for Dip Count
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['DipCount'], color='lightblue', width=0.5)
plt.title('Boxplot of Dip Counts')
plt.xlabel('Dip Count')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Apply Log Transformation to Dip Count (Add 1 to avoid log(0))
df['LogDipCount'] = np.log1p(df['DipCount'])

# Visualization 2: Dip Count Over Time (Log Scale)
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['LogDipCount'], marker='o', color='purple', label='Log Dip Count')
plt.title('Dip Count Over Time (Log Scale)')
plt.xlabel('Date')
plt.ylabel('Log Dip Count')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Visualization 3: Temperature vs. Log Dip Count
plt.figure(figsize=(8, 6))
plt.scatter(df['Temperature'], df['LogDipCount'], color='blue', edgecolor='black', alpha=0.7)
plt.title('Temperature vs. Log Dip Count')
plt.xlabel('Temperature (°C)')
plt.ylabel('Log Dip Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization 4: Cloud Coverage vs. Log Dip Count
plt.figure(figsize=(8, 6))
plt.scatter(df['CloudCoverage'], df['LogDipCount'], color='green', edgecolor='black', alpha=0.7)
plt.title('Cloud Coverage vs. Log Dip Count')
plt.xlabel('Cloud Coverage (%)')
plt.ylabel('Log Dip Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualization 5: Breeding Frequency Per Site (No Log Transformation)
breeding_frequency = df[df['DipCount'] > 0.1].groupby('SiteID')['DipCount'].count()
plt.figure(figsize=(10, 6))
breeding_frequency.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Breeding Frequency Per Site')
plt.xlabel('Site ID')
plt.ylabel('Number of Days Breeding (Dip Count > 0.1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 6: Correlation Heatmap (With Log Dip Count)
correlation_data = df[['Temperature', 'Precipitation', 'CloudCoverage', 'LogDipCount']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Using Log Dip Count)')
plt.tight_layout()
plt.show()
