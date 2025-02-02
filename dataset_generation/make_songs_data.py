import kagglehub
import pandas as pd
import os
import pickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import re


def extract_year(date):
    """Extracts the year from different date formats"""
    match = re.search(r'\d{4}', str(date))
    return match.group(0) if match else None



# Load Spotify tracks dataset
file_path = (r"C:\Users\amitb\OneDrive - Technion\עידן ועמית- למידה עמוקה\projecton\datasets\spotify\tracks.csv")
spotify_tracks = pd.read_csv(file_path, usecols=['name', 'artists', 'release_date', 'popularity'])

# Drop duplicates with same name and artist
spotify_tracks = spotify_tracks.drop_duplicates(subset=['name', 'artists'], keep='last').reset_index(drop=True)

# Process the 'artists' field
spotify_tracks['artists'] = spotify_tracks['artists'].apply(lambda a: a.split(",")[0].replace("['", "").replace("']", "").replace("'", ""))

# Ensure release_date is only the year
spotify_tracks['release_date'] = spotify_tracks['release_date'].apply(extract_year)

# Apply popularity classification
spotify_tracks['popularity'] = spotify_tracks['popularity']

# Load spectrogram file details
filename_list = []
dirname_list = []
artistname_list = []
songname_list = []

print(len(spotify_tracks))

# Save as .pkl file
output_file = r"C:\Users\amitb\OneDrive - Technion\עידן ועמית- למידה עמוקה\projecton\datasets\spotify_data.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(spotify_tracks, f)

print(f"Dataset saved to {output_file}, containing {len(spotify_tracks)} records.")

