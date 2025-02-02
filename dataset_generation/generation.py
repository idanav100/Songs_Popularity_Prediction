# Script for making custom dataset
# Please change the corresponding paths:
# spotify_file_path - this is the path for csv file including 500k+ song's features.
# spectrogram_dir - this is the path for the folder where the downsamples spectrograms appear.
# output file - path for output pickle file

import pandas as pd
import os
import pickle
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import re

# ----------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------

def extract_year(date):
    """
    Extracts the year from different date formats.

    Args:
        date (str): The release date (could be in various formats).

    Returns:
        str: The year extracted from the date string or None if no valid year is found.
    """
    match = re.search(r'\d{4}', str(date))
    return match.group(0) if match else None


def load_image(row):
    """
    Loads an image from the path specified in the row.

    Args:
        row (pd.Series): A row from the DataFrame containing information about the spectrogram.

    Returns:
        tuple: Index of the row and the loaded image, or None if the image is not found.
    """
    img_path = os.path.join(row['directory'], row['spectro_filename'])
    if os.path.exists(img_path):
        return row.name, Image.open(img_path)  # row.name = index
    else:
        print(f"Warning: Image {img_path} not found!")
        return row.name, None


# ----------------------------------------------------------------------------------------------------------------
# Load Spotify Tracks Data
# ----------------------------------------------------------------------------------------------------------------

def load_spotify_data(file_path):
    """
    Loads the Spotify tracks dataset from a CSV file and processes it.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame containing track information.
    """
    spotify_tracks = pd.read_csv(file_path, usecols=['name', 'artists', 'release_date', 'popularity'])

    # Drop duplicates based on name and artist
    spotify_tracks = spotify_tracks.drop_duplicates(subset=['name', 'artists'], keep='last').reset_index(drop=True)

    # Process 'artists' field to remove extra characters
    spotify_tracks['artists'] = spotify_tracks['artists'].apply(
        lambda a: a.split(",")[0].replace("['", "").replace("']", "").replace("'", ""))

    # Ensure release_date contains only the year
    spotify_tracks['release_date'] = spotify_tracks['release_date'].apply(extract_year)

    return spotify_tracks


# ----------------------------------------------------------------------------------------------------------------
# Load Spectrogram Data
# ----------------------------------------------------------------------------------------------------------------

def load_spectrogram_data(spectrogram_dir):
    """
    Loads information about spectrogram images, including artist and song names.

    Args:
        spectrogram_dir (str): Directory containing spectrogram files.

    Returns:
        pd.DataFrame: DataFrame containing spectrogram details like artist, song, filename, and directory.
    """
    filename_list = []
    dirname_list = []
    artistname_list = []
    songname_list = []

    for dirname, _, filenames in os.walk(spectrogram_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                filename_list.append(filename)
                dirname_list.append(dirname)
                artistname_list.append(filename.split('_-_')[0].replace("_", " "))
                songname_list.append(filename.split('_-_')[1].split('.png')[0].replace("_", " "))

    # Create DataFrame for spectrograms
    spectro_df = pd.DataFrame(list(zip(artistname_list, songname_list, filename_list, dirname_list)),
                              columns=['artists', 'name', 'spectro_filename', 'directory'])

    return spectro_df


# ----------------------------------------------------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------------------------------------------------

def main():
    """
    Main function to run the script and process Spotify and spectrogram data.
    """
    # Define file paths - add your own path here
    spotify_file_path = r"datasets\spotify\tracks.csv"
    spectrogram_dir = r"datasets\spectograms\dataset_down_sampled"
    output_file = r"processed_data_regression.pkl"

    # Load Spotify data
    spotify_tracks = load_spotify_data(spotify_file_path)

    # Load spectrogram data
    spectro_df = load_spectrogram_data(spectrogram_dir)

    # Merge Spotify data with spectrogram data on artist and song name
    merged_df = spectro_df.merge(
        spotify_tracks[['artists', 'name', 'release_date', 'popularity']],
        on=['artists', 'name'],
        how='left'
    ).dropna(subset=['popularity', 'release_date'])

    # Remove duplicates
    merged_df = merged_df.drop_duplicates(subset=['artists', 'name'], keep='last').reset_index(drop=True)

    # Load spectrogram images in parallel
    spectrogram_data = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(load_image, [row for _, row in merged_df.iterrows()])
        spectrogram_data = {index: img for index, img in results if img is not None}

    # Combine data for saving
    final_data = []
    for index, row in merged_df.iterrows():
        spectrogram = spectrogram_data.get(index, None)
        if spectrogram is not None:
            final_data.append({
                'name': row['name'],
                'artist': row['artists'],
                'release_date': row['release_date'],
                'spectrogram': spectrogram,
                'popularity': row['popularity']
            })

    # Save the final data as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(final_data, f)

    print(f"Dataset saved to {output_file}, containing {len(final_data)} records.")


# ----------------------------------------------------------------------------------------------------------------
# Run the Script
# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
