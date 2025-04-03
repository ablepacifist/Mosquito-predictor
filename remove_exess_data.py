#!/usr/bin/env python3
import pandas as pd
import sys

def main(input_file, output_file):
    # Load the CSV file into a DataFrame.
    # Using dtype=str to keep everything as text (preventing accidental type conversion).
    df = pd.read_csv(input_file, dtype=str)
    
    # Replace any NaNs with empty strings.
    df.fillna('', inplace=True)
    
    # Check that the relevant columns exist.
    required_cols = ['Temperature', 'Action']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Expected column '{col}' not found in CSV.")
            sys.exit(1)
    
    # Create a mask to filter rows.
    # We want to remove rows where the Temperature is missing (empty or whitespace)
    # AND the Action equals "none" (case-insensitive).
    mask = ~((df['Temperature'].str.strip() == '') & (df['Action'].str.lower() == 'none'))
    
    df_filtered = df[mask]
    
    # Optionally, print out how many rows were removed.
    removed = len(df) - len(df_filtered)
    print(f"Removed {removed} rows. Saving {len(df_filtered)} remaining rows to {output_file}.")
    
    # Write the filtered DataFrame to a new CSV file.
    df_filtered.to_csv(output_file, index=False)
    
if __name__ == '__main__':
    input_csv = "combined_data.csv"       # Update to your actual input file path if needed.
    output_csv = "combined_data_filtered.csv"  # Output file.
    main(input_csv, output_csv)
