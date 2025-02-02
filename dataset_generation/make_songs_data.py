import kagglehub
import pandas as pd
import os
import pickle
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

def load_spotify_data(file_path):
    """
    Loads and processes Spotify track data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame containing track information.
    """
    # Load only the necessary columns
    spotify_tracks = pd.read_csv(file_path, usecols=['name', 'artists', 'release_date', 'popularity'])

    # Drop duplicates based on name and artist
    spotify_tracks = spotify_tracks.drop_duplicates(subset=['name', 'artists'], keep='last').reset_index(drop=True)

    # Process 'artists' field to remove extra characters and keep only the first listed artist
    spotify_tracks['artists'] = spotify_tracks['artists'].apply(
        lambda a: a.split(",")[0].replace("['", "").replace("']", "").replace("'", "")
    )

    # Convert release_date to only the year
    spotify_tracks['release_date'] = spotify_tracks['release_date'].apply(extract_year)

    # Keep popularity as is; can do classification or leave numeric
    return spotify_tracks


# ----------------------------------------------------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------------------------------------------------

def main():
    """
    Main function to load Spotify data and save it as a pickle file.
    """
    # Define file paths
    spotify_file_path = r"C:\Users\amitb\OneDrive - Technion\עידן ועמית- למידה עמוקה\projecton\datasets\spotify\tracks.csv"
    output_file = r"C:\Users\amitb\OneDrive - Technion\עידן ועמית- למידה עמוקה\projecton\datasets\spotify_data.pkl"

    # Load and process Spotify tracks
    spotify_tracks = load_spotify_data(spotify_file_path)

    # Print dataset size for reference
    print(f"Number of Spotify tracks after cleaning: {len(spotify_tracks)}")

    # Save the final data as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(spotify_tracks, f)

    print(f"Dataset saved to {output_file}, containing {len(spotify_tracks)} records.")


# ----------------------------------------------------------------------------------------------------------------
# Run the Script
# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
