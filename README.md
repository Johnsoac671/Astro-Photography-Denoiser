# Astro-Photography Denoiser & Upscaler

This project explores the impact of image denoising and super-resolution (upscaling) on the classification of astronomical objects. It implements a Convolutional Neural Network (CNN) to upscale and denoise low-resolution galaxy images and evaluates the performance using a Random Forest classifier.

## Features

* **Automated Data Collection**: Scripts to query and download FITS (Flexible Image Transport System) files from the SkyView API.
* **Deep Learning Upscaler**: A custom CNN built with PyTorch to upscale 64x64 images to 256x256, utilizing a combined L1/MSE loss function.
* **Image Processing Pipeline**: Utilities for normalizing FITS data, adding artificial noise/corruption, and automatically aligning image pairs using phase correlation.
* **Classification Evaluation**: A Random Forest classifier framework to benchmark different upscaling methods (CNN vs. Nearest/Bilinear/Bicubic/Lanczos) based on downstream classification accuracy.

## Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install numpy pandas torch opencv-python scikit-learn astropy requests matplotlib
```

## Project Structure

* **`Astroquery_queryer.py`**: Fetches raw FITS images from NASA's SkyView API based on coordinates in a CSV.
* **`upscaler.py`**: Contains the CNN model definition (`CnnUpscaler`), custom loss function, and the training loop for the super-resolution model.
* **`classifier.py`**: The evaluation script. It loads images, applies the selected upscaling method, extracts features, and runs a Random Forest classifier to report accuracy metrics.
* **`datasets/`**: Directory containing the input CSVs (e.g., `galaxies_categorical.csv`) with galaxy coordinates and labels.

## Usage

### 1. Data Acquisition
Download galaxy images using the query script. This reads from `datasets/galaxies_categorical.csv` and downloads FITS files into the chosen directory. 

*Note: the galaxies_.csv files contain hundreds of thousands of galaxies, so you will likely want to end the downloading early.*

```bash
python "Astroquery_queryer.py"
```

### 2. Training the Upscaler (Optional)
To train the CNN denoiser/upscaler from scratch. This script handles data loading, noise injection, and training loops. It saves model checkpoints to disk.

```bash
python upscaler.py
```
*Note: You can modify constants in `upscaler.py` (e.g., `BATCH_SIZE`, `NOISE_CORRUPTION_RATE`) to tweak training parameters.*

### 3. Running the Classifier & Evaluation
Evaluate how well the upscaler helps in classifying galaxies. You can switch between different modes (`cnn`, `bicubic`, `bilinear`, etc.) to see the difference in accuracy.

```bash
python classifier.py
```

**Key Configuration in `classifier.py`:**
* `DATA_MODE`: Set this to `"cnn"` to use your trained model, or `"bicubic"`, `"bilinear"`, etc., for baselines.
* `UPSCALER_CHECKPOINT`: Path to your trained `.pth` model file (required for CNN mode).
* `FEATURE_LABEL`: The target classification label (e.g., `"edgeon"`).

## Methodology

1.  **Data Prep**: Images are normalized and optionally corrupted with noise.
2.  **Upscaling**:
    * **Baseline**: Images are upscaled using standard OpenCV interpolation.
    * **CNN**: Images are passed through the PyTorch model trained to recover high-frequency details.
3.  **Classification**: The processed images are flattened and fed into a Random Forest Classifier.
4.  **Metrics**: The script reports Confusion Matrices, F1-scores, and Balanced Accuracy.
