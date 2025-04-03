import requests
import csv
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_KEY = 'ee40d5eff7b14962a31172700250204'  # Your actual World Weather Online API key
BASE_URL = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx'
LOCATION = 'Sunfish Lake, MN'
INPUT_CSV = 'data_1.csv'
OUTPUT_CSV = 'combined_data.csv'

# --- WEATHER FETCH FUNCTION ---
def fetch_weather_for_day(location, date_str, status_tracker):
    """
    Fetch weather data for a given location and date (YYYY-MM-DD format).
    Stops all data retrieval if two consecutive 429 responses are encountered globally.
    Returns a dictionary with keys Temperature, Precipitation, and CloudCoverage.
    If data is unavailable or status_tracker signals halt, returns empty strings.
    """
    if status_tracker['halt']:
        print("Global halt signal received. Skipping weather data retrieval.")
        return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
    
    params = {
        'key': API_KEY,
        'q': location,
        'date': date_str,
        'enddate': date_str,
        'format': 'json',
        'tp': '24'  # daily data
    }
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        status_tracker['consecutive_429_count'] = 0  # Reset the counter on a successful request
        data = response.json()
        try:
            if 'weather' not in data.get('data', {}) or not data['data']['weather']:
                print(f"No weather data returned for {date_str}.")
                return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
            
            weather_day = data['data']['weather'][0]
            avg_temp = weather_day.get('avgtempC', '')
            precip = weather_day.get('precipMM', '')
            # Use the first hourly record if available for cloud coverage
            if 'hourly' in weather_day and weather_day['hourly']:
                cloud = weather_day['hourly'][0].get('cloudcover', '')
            else:
                cloud = ''
            return {'Temperature': avg_temp, 'Precipitation': precip, 'CloudCoverage': cloud}
        except Exception as e:
            print(f"Error parsing weather data for {date_str}: {e}")
            return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
    elif response.status_code == 429:
        status_tracker['consecutive_429_count'] += 1
        print(f"Received status 429 for {date_str}. Consecutive 429 count: {status_tracker['consecutive_429_count']}")
        if status_tracker['consecutive_429_count'] >= 2:
            print("Two consecutive status 429 errors encountered. Halting all data retrieval.")
            status_tracker['halt'] = True  # Signal a global halt
        return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
    else:
        print(f"Error fetching weather for {date_str}, status code: {response.status_code}")
        return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}

# --- GENERATE NEW ROWS FROM data_1.csv ---
def generate_new_rows(input_csv, location, fetch_weather=True):
    """
    Reads the input raw CSV and generates a five-day sequence (the recorded day plus the four previous days)
    for each row. Weather fetches are cached so that the same date is not repeatedly requested.
    Returns a tuple (fieldnames, new_rows).
    """
    new_rows = []
    status_tracker = {'consecutive_429_count': 0, 'halt': False}
    
    # Initialize a weather fetch cache: keys will be date strings ('YYYY-MM-DD')
    weather_cache = {}
    
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        # Append weather columns to original fieldnames if they are not already there.
        fieldnames = reader.fieldnames + ['Temperature', 'Precipitation', 'CloudCoverage']
        
        for row in reader:
            if status_tracker['halt']:
                print("Global halt signal received. Stopping row generation.")
                break
            
            try:
                recorded_date = datetime.strptime(row['Date'], '%m/%d/%Y')
            except Exception as e:
                print(f"Error parsing date '{row['Date']}' in row: {e}")
                continue
            
            # For each row, generate a five-day sequence: preceding 4 days + recorded day.
            for i in range(5):
                current_date = recorded_date - timedelta(days=(4 - i))
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Check the cache first.
                if fetch_weather:
                    if date_str in weather_cache:
                        weather = weather_cache[date_str]
                    else:
                        weather = fetch_weather_for_day(location, date_str, status_tracker)
                        weather_cache[date_str] = weather
                else:
                    weather = {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
                
                # Create a new row based on the original row.
                new_row = row.copy()
                if current_date != recorded_date:
                    new_row['TreatedBy'] = 'none'
                    new_row['Date'] = current_date.strftime('%m/%d/%Y')
                    new_row['Wetness'] = '-1'
                    new_row['Action'] = 'none'
                    new_row['DipCount'] = ''       # Clear treatment-specific data
                    new_row['Material'] = ''
                else:
                    new_row['Date'] = recorded_date.strftime('%m/%d/%Y')
                
                # Append weather data.
                new_row['Temperature'] = weather.get('Temperature', '')
                new_row['Precipitation'] = weather.get('Precipitation', '')
                new_row['CloudCoverage'] = weather.get('CloudCoverage', '')
                
                new_rows.append(new_row)
    
    return fieldnames, new_rows


# --- MERGE EXISTING combined_data.csv WITH NEW ROWS ---
def merge_csvs(existing_file, new_fieldnames, new_rows):
    """
    Reads the existing CSV and merges it with new_rows based on composite key (SiteID, Date).
    Existing rows are not overwritten. Filters out invalid fields.
    """
    existing_rows = []
    if os.path.exists(existing_file):
        with open(existing_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            existing_rows = list(reader)
    
    # Create a dictionary for merging using (SiteID, Date) as the key
    merged_dict = {}
    for row in existing_rows:
        key = (row.get('SiteID', ''), row.get('Date', ''))
        merged_dict[key] = row
    for row in new_rows:
        key = (row.get('SiteID', ''), row.get('Date', ''))
        if key not in merged_dict:
            merged_dict[key] = row
    
    # Prepare filtered rows
    merged_rows = []
    for row in merged_dict.values():
        # Remove invalid keys like None
        if None in row:
            del row[None]
        # Filter to only include valid keys
        filtered_row = {key: row.get(key, '') for key in new_fieldnames}
        merged_rows.append(filtered_row)
    
    # Write to the CSV file
    with open(existing_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)


# --- MAIN SCRIPT ---
if __name__ == '__main__':
    answer = input("Do you want to fetch new weather data? (Y/N): ").strip().lower()
    if answer == 'y':
        fetch_flag = True
        print("Fetching weather data for missing entries...")
    else:
        fetch_flag = False
        print("Not fetching new weather data; merging without updating weather fields.")
    
    # Generate rows from raw data (data_1.csv); set fetch_weather flag according to user answer
    fieldnames, new_rows = generate_new_rows(INPUT_CSV, LOCATION, fetch_weather=fetch_flag)
    
    # Merge with existing combined_data.csv (or create it if it doesn't exist)
    merge_csvs(OUTPUT_CSV, fieldnames, new_rows)
    
    print(f"Combined CSV file written to {OUTPUT_CSV}")
