# Dust Powerloss Detection

This repository contains a deep learning and machine learning project for detecting whether a solar panel image is clean or dusty. The goal is to support predictive maintenance by identifying dust accumulation on photovoltaic panels, which can reduce power generation efficiency and increase maintenance costs.

The project combines:
- image preprocessing and feature engineering,
- exploratory data analysis,
- CNN-based transfer learning,
- classical machine learning classifiers on extracted features,
- and a small Flask application for image-based inference.

The work is structured as a research and prototype project, and it is designed to help evaluate which model performs best for solar panel cleanliness classification.

---

## Project purpose and function

The main function of this repository is to classify solar panel images into two categories:
- Clean
- Dusty

This is a binary image classification problem. Dust on solar panels blocks sunlight and reduces energy output, so automatically detecting contamination can help schedule cleaning operations before efficiency drops significantly.

In practical terms, the repository:
1. loads solar panel images from a labeled dataset,
2. preprocesses and normalizes them,
3. explores the dataset to inspect visual patterns,
4. compares deep feature extraction models such as MobileNet, InceptionV3, and VGG16,
5. trains classical classifiers such as Logistic Regression, Decision Tree, and related models on extracted features,
6. evaluates performance using accuracy and related metrics,
7. exposes a simple web prediction interface using Flask.

This makes the repository useful both as a study project and as a starting point for a real-world maintenance-support system.

---

## Repository structure

The project contains a set of Jupyter notebooks and one Flask API script.

- `Solar Panel Dust Detection Notebook.ipynb`  
  Main research notebook that introduces the project, loads the dataset, performs EDA, and explores the full dust detection workflow.

- `Feature Engineering.ipynb`  
  Focuses on cleaning and preparing the solar panel images, handling white-background detection, splitting data into training and test groups, and improving image quality for model training.

- `Image Augmentation Notebook.ipynb`  
  Applies augmentation techniques such as rotation and horizontal flipping to improve generalization.

- `CNN + ML Models.ipynb`  
  Builds a CNN-based feature extractor and combines it with machine learning models.

- `Inception + ML Models.ipynb`  
  Uses InceptionV3 as a feature extractor with traditional ML models for classification.

- `MobileNet + ML Models.ipynb`  
  Uses MobileNet for transfer learning and evaluates feature-extractor + classifier pipelines.

- `VGG16 + ML Models.ipynb`  
  Similar workflow using VGG16 as the feature extractor.

- `cnn_flask_api.py`  
  Flask application that loads a trained model and classifies uploaded solar panel images as dusty or clean.

- `LICENSE`  
  Project license.

- `README.md`  
  Project documentation.

---

## Dataset and expected folder layout

The notebooks expect the dataset to be organized in a local folder named `Detect_solar_dust/` with two classes:

```text
Detect_solar_dust/
├── Clean/
├── Dusty/
└── (optionally Train/Test folders created during feature engineering)
```

The project references the Kaggle dataset for solar panel dust detection and uses labeled images for clean and dusty panels. The dataset is not bundled inside this repository, so it must be downloaded and placed locally before running the notebooks.

The code in the notebooks uses the directory names `Clean` and `Dusty`, and later creates training and testing subsets for model evaluation.

---

## Core workflow in the repository

The repository follows a standard deep-learning experimentation workflow:

1. Import the required libraries
   - TensorFlow
   - OpenCV
   - NumPy
   - Matplotlib
   - Pandas
   - scikit-learn
   - tqdm

2. Define image paths and constants
   - image size: 224 x 224
   - batch size
   - random seed
   - grayscale flag

3. Load the solar panel images from the `Clean` and `Dusty` directories

4. Convert image arrays into a format suitable for model training

5. Label data
   - clean panels = 0
   - dusty panels = 1

6. Split data into train, validation, and test sets

7. Normalize pixel values to the range [0, 1]

8. Use feature extraction models such as:
   - MobileNet
   - InceptionV3
   - VGG16

9. Pass extracted features to machine learning classifiers such as Logistic Regression and Decision Tree

10. Evaluate using metrics like accuracy, confusion patterns, and ROC-AUC related reporting

11. Save the model and deploy it through a Flask app for inference on new images

---

## Why this repository matters

Solar panels are used widely for renewable energy generation, but dust accumulation can reduce their efficiency. Manual inspection is time-consuming and expensive, especially for large solar farms or industrial installations.

This project demonstrates how computer vision and machine learning can reduce that burden by automatically classifying solar panel condition from images. Instead of relying only on human inspection, the system can provide fast, repeatable, scalable screening.

This is especially relevant for:
- solar farms,
- rooftop installations,
- industrial maintenance teams,
- remote monitoring systems,
- and energy-efficiency audits.

---

## Requirements

Use a Python environment with the packages below installed.

```bash
python --version
```

Recommended: Python 3.9 to 3.11.

Install dependencies:

```bash
pip install tensorflow opencv-python matplotlib pandas numpy scikit-learn imbalanced-learn tqdm
```

If you are using an NVIDIA GPU, also make sure your TensorFlow version matches your CUDA setup.

---

## How to run the repository

### 1. Clone the project

```bash
git clone <repository-url>
cd Dust_Powerloss_Detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install tensorflow opencv-python matplotlib pandas numpy scikit-learn imbalanced-learn tqdm
```

### 4. Download the dataset

Place the dataset in the project root with this structure:

```text
Dust_Powerloss_Detection/
├── Detect_solar_dust/
│   ├── Clean/
│   └── Dusty/
├── notebooks...
├── cnn_flask_api.py
└── README.md
```

The notebooks refer to `Detect_solar_dust/`, so this folder name must match exactly.

### 5. Run the notebooks

Open the notebook you want to work with, for example:

- `Solar Panel Dust Detection Notebook.ipynb`
- `MobileNet + ML Models.ipynb`
- `CNN + ML Models.ipynb`

Then run the cells in order. These notebooks will:
- load the image data,
- preprocess the data,
- split it into train/test sets,
- extract features,
- train the models,
- print metrics,
- and allow experimentation with different architectures.

### 6. Save a trained model for the web app

The Flask file loads a model from:

```python
model = load_model("Models/Mobilenet.h5")
```

That means you must create a `Models/` folder and save the trained model there as:

```text
Models/Mobilenet.h5
```

If this file is missing, the Flask application will fail to start.

---

## How to run the Flask prediction app

The repository also contains a small web app for upload-based prediction.

### Important note

This project repository is partly experimental and does not include the required model file or HTML templates by default. The application expects:

```text
Models/
└── Mobilenet.h5

templates/
├── home.html
└── result.html
```

The script `cnn_flask_api.py` will render the homepage and then accept an uploaded image and predict whether it is dusty or clean.

### Start the app

```bash
python cnn_flask_api.py
```

Then open in a browser:

```text
http://127.0.0.1:5000/
```

Upload an image of a solar panel and the app will return a classification result:
- Dusty
- Clean

---

## What the model is doing

The classification logic is based on image features extracted by convolutional neural networks. These models are excellent at identifying patterns such as:
- panel texture,
- dust distribution,
- background contrast,
- lighting variations,
- and surface contamination.

The repository uses transfer learning to take advantage of models pretrained on ImageNet and repurpose them for this solar-panel classification task. Instead of training a CNN from scratch, the notebooks use pretrained architectures as feature extractors and then feed those features into simpler classifiers.

This approach is effective when the dataset is limited or when faster experimentation is required.

---

## Model comparison included in the project

The notebooks compare several CNN-based approaches:

- CNN + traditional classifier
- MobileNet + ML models
- InceptionV3 + ML models
- VGG16 + ML models

The purpose is to evaluate which architecture extracts the most useful features for distinguishing clean and dusty solar panels. This is a common benchmarking setup in machine learning research and is exactly what this repository is designed to explore.

---

## Practical use case

This system can be used as a decision-support tool in solar maintenance operations:

- detect when panels need cleaning,
- reduce unexpected energy loss,
- improve inspection efficiency,
- support preventive maintenance planning,
- and contribute to higher renewable energy reliability.

---

## Limitations and notes

This repository is best understood as an educational and experimental project rather than a production-ready deployment package.

Some important points:
- The dataset is external and must be downloaded separately.
- The trained model file is not included in the repository.
- The HTML templates for the Flask app are not present in the repo snapshot.
- Training and evaluation are done in notebooks rather than a packaged application.
- The final deployment part is a prototype interface, not a full production web service.

These limitations are common in academic ML repositories and are easy to resolve by adding the dataset, training the model, and providing the required templates.

---

## Summary

This repository is a complete exploration of solar panel dust detection using computer vision and machine learning. It demonstrates how image-based classification can detect dust accumulation, compare several transfer-learning models, and turn the best-performing model into a simple prediction interface.

In short, the repository’s function is to train and evaluate models that classify solar panel images as clean or dusty, and to provide a lightweight Flask-based demonstration of that classification process.

---

## Recommended next steps

1. Download and place the Kaggle solar-panel dust dataset in the `Detect_solar_dust/` folder.
2. Open the main notebook and run all cells sequentially.
3. Compare the model variants and identify the best-performing architecture.
4. Save the best model to `Models/Mobilenet.h5`.
5. Add the required HTML templates and launch the Flask app.

This will turn the repository from a research notebook collection into a usable image-classification demo.
