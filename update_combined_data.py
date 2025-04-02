import csv
import requests
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# --- CONFIGURATION ---
API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your World Weather Online API key
BASE_URL = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx'
LOCATION = 'Sunfish Lake, MN'
CSV_FILE = 'combined_data.csv'  # This file will be updated in-place

# --- FUNCTION TO FETCH WEATHER DATA FOR A SINGLE DAY ---
def fetch_weather_for_day(location, date_str):
    """
    Fetch weather data for a given location and date (in 'YYYY-MM-DD' format).
    Returns a dict with keys: Temperature, Precipitation, CloudCoverage.
    If the API does not return data, returns empty strings for those keys.
    """
    params = {
        'key': API_KEY,
        'q': location,
        'date': date_str,
        'enddate': date_str,
        'format': 'json',
        'tp': '24'  # data at daily intervals
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            weather_list = data.get('data', {}).get('weather', [])
            if not weather_list:
                print(f"No weather data returned for {date_str}.")
                return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
            weather_day = weather_list[0]
            avg_temp = weather_day.get('avgtempC', '')
            precip = weather_day.get('precipMM', '')
            if 'hourly' in weather_day and len(weather_day['hourly']) > 0:
                cloud = weather_day['hourly'][0].get('cloudcover', '')
            else:
                cloud = ''
            return {
                'Temperature': avg_temp,
                'Precipitation': precip,
                'CloudCoverage': cloud
            }
        except Exception as e:
            print(f"Error parsing weather data for {date_str}: {e}")
            return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}
    else:
        print(f"Error fetching weather for {date_str}, status code: {response.status_code}")
        return {'Temperature': '', 'Precipitation': '', 'CloudCoverage': ''}

# --- FUNCTION TO PROCESS THE CSV AND FILL MISSING DATA ---
def process_missing_data(csv_filename, location):
    updated_rows = []
    
    # Read the existing CSV file.
    with open(csv_filename, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames  # Keep same columns
    
        for row in reader:
            # Check if Temperature is missing. (Assuming missing means an empty string or only whitespace.)
            if not row['Temperature'].strip():
                try:
                    dt = datetime.strptime(row['Date'], '%m/%d/%Y')
                    date_api = dt.strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"Error parsing date '{row['Date']}': {e}")
                    date_api = row['Date']  # fallback
                
                weather = fetch_weather_for_day(location, date_api)
                row['Temperature'] = weather.get('Temperature', '')
                row['Precipitation'] = weather.get('Precipitation', '')
                row['CloudCoverage'] = weather.get('CloudCoverage', '')
            updated_rows.append(row)
    
    # Write updated data back to the same file (overwrite it)
    with open(csv_filename, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    print(f"CSV file '{csv_filename}' updated at {datetime.now()}")

# --- JOB FUNCTION FOR THE SCHEDULER ---
def scheduled_job():
    print(f"Scheduled job started at {datetime.now()}")
    process_missing_data(CSV_FILE, LOCATION)
    print(f"Scheduled job completed at {datetime.now()}")

# --- SCHEDULER SETUP ---
if __name__ == '__main__':
    scheduler = BlockingScheduler()
    # Schedule the job to run every 25 hours.
    scheduler.add_job(scheduled_job, 'interval', hours=25)
    print("Scheduler started. The job will run every 25 hours to update the CSV file.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
