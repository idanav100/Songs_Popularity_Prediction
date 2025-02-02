# Script for Downsampling spectrograms from billboard-hot-100-19602020-spectrograms.
# for kaggle mode:
#   install for running the script using Kaggle mode:
#   pip install kaggle
#   download the key API from kaggle's user and locate it on KAGGLE_CONFIG_DIR
#   change WORK_MODE to 'Kaggle'
# for Manual mode:
# change the SOURCE_FOLDER,DESTINATION_FOLDER as you wish

import os
import cv2  # OpenCV for image processing
import numpy as np
from scipy.signal import resample
from kaggle.api.kaggle_api_extended import KaggleApi

# ----------------------------------------------------------------------------------------------------------------
# Constants and Configuration
# ----------------------------------------------------------------------------------------------------------------

WORK_MODE = 'Manual'  # Set work mode: 'kaggle' or 'Manual'
SOURCE_FOLDER = r"..\spectrograms"
DESTINATION_FOLDER = r"..\down_sampled_spectrograms"
DATASET = 'tpapp157/billboard-hot-100-19602020-spectrograms'
TARGET_DIR = 'dataset_down_sampled'
KAGGLE_CONFIG_DIR = 'kaggle.json file location'

# ----------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------

def downsample_spectrogram(spectrogram, target_width=1024, target_height=256):
    """
    Downsample the input spectrogram to a target size using resampling.
    """
    # Step 1: Resample along the time axis (columns)
    resampled_time = resample(spectrogram, target_width, axis=1)

    # Step 2: Resize along the frequency axis (rows)
    resampled_spectrogram = resample(resampled_time, target_height, axis=0)

    # Convert back to numpy array and return
    resized_spectrogram = np.array(resampled_spectrogram)
    return resized_spectrogram


def download_kaggle_dataset():
    """
    Downloads the dataset from Kaggle and extracts it to the target directory.
    """
    current_dir = os.getcwd()  # Get current working directory

    # Define the relative path to the Kaggle config directory
    KAGGLE_CONFIG_DIR = os.path.join(current_dir, 'dataset_generation')
    os.environ['KAGGLE_CONFIG_DIR'] = KAGGLE_CONFIG_DIR

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=TARGET_DIR, unzip=True)
    print(f"Dataset downloaded and extracted to {TARGET_DIR}")


def process_images(source_folder, destination_folder):
    """
    Processes all spectrogram images, downsamples them, and saves to the destination folder.
    """
    os.makedirs(destination_folder, exist_ok=True)

    # Process each sub-folder in sorted order
    for year_folder in sorted(os.listdir(source_folder)):
        source_path = os.path.join(source_folder, year_folder)
        dest_path = os.path.join(destination_folder, year_folder)

        if os.path.isdir(source_path):  # Ensure it's a directory
            os.makedirs(dest_path, exist_ok=True)  # Create corresponding sub-folder in destination

            image_files = sorted([f for f in os.listdir(source_path) if f.endswith(".png")])  # Sort files

            # Process each image
            for file_name in image_files:
                image_path = os.path.join(source_path, file_name)
                output_path = os.path.join(dest_path, file_name)

                # Read the image
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Error reading {image_path}")
                    continue

                image = image.astype(np.float32)

                # Downsample the spectrogram
                corrected_image = downsample_spectrogram(image, target_width=1024, target_height=256)

                # Normalize and convert back to uint8
                corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
                corrected_image = corrected_image.astype(np.uint8)

                # Save the corrected image
                cv2.imwrite(output_path, corrected_image)

            print(f"Processed and saved corrected images in {dest_path}")

    print("All images have been processed and saved.")


# ----------------------------------------------------------------------------------------------------------------
# Main Script Logic
# ----------------------------------------------------------------------------------------------------------------

def main():
    """
    Main function to run the script based on the selected work mode.
    """
    print(f"Work mode selected: {WORK_MODE}")  # Debugging line to check the work mode

    if WORK_MODE == 'kaggle':
        # Download the dataset from Kaggle
        print("Entering Kaggle mode...")  # Debugging line
        download_kaggle_dataset()
        # Set the source folder after downloading
        source_folder = os.path.join(TARGET_DIR, 'spectrograms')
    else:
        print("Entering Manual mode...")  # Debugging line
        source_folder = SOURCE_FOLDER  # Use the manual source folder

    # Process and downsample images
    process_images(source_folder, DESTINATION_FOLDER)


# ----------------------------------------------------------------------------------------------------------------
# Run the Script
# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
