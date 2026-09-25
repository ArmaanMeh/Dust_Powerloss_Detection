# Dust Powerloss Detection: Development Blog (Weeks 6–9)

## Project overview

I started this project with a clear goal: build a solar panel dust detection system that can distinguish between clean and dusty panels using image-processing and deep learning techniques. Over the four weeks, I worked through the full repository and connected the different notebooks into one continuous development story: the dataset exploration in the main notebook, image cleanup in the feature engineering notebook, augmentation and CNN feature extraction, transfer learning with MobileNet, Inception, and VGG16, and finally the deployment layer in the Flask application.

I used the repository as a complete pipeline rather than as isolated experiments. The notebooks and API script all contribute to one workflow: preprocess images, train feature extractors, compare classifiers, tune the best model, and deploy it for prediction. This blog is written in first person and summarises exactly how I moved through the project during Weeks 6 to 9.

## Repository coverage map

I covered the entire repository across the four weeks:

- Week 6: README.md, Solar Panel Dust Detection Notebook.ipynb, Feature Engineering.ipynb
- Week 7: Image Augmentation Notebook.ipynb, CNN + ML Models.ipynb
- Week 8: Inception + ML Models.ipynb, MobileNet + ML Models.ipynb, VGG16 + ML Models.ipynb
- Week 9: cnn_flask_api.py, README.md, model deployment and evaluation workflow

---

## Week 6: Data understanding, preprocessing, and feature engineering

I began the week by revisiting the project goal and understanding the repository structure. The README gave the full direction of the system: identify whether a solar panel image is clean or dusty, prepare the image dataset, extract visual features, and evaluate whether the system can support maintenance decisions for solar energy systems. I treated the repository as a full machine-learning pipeline rather than just a single notebook.

The first major step was preparing the dataset. I loaded the images from the Detect_solar_dust folder and checked the class distributions. The project uses a binary classification setup with Clean and Dusty classes, so each image needed to be labelled correctly before any model training could begin. This part of the work was grounded in the Solar Panel Dust Detection Notebook, which focuses on the overall architecture of the problem, EDA, and the initial data-loading strategy.

I also worked through the feature engineering notebook, where I handled the cleaning and preparation of images before model training. This is a very important phase because solar panel imagery often contains distractions such as white backgrounds, extra text, or human interference. I used CV-based functions to remove white-background images, crop the solar panel region, and organise the dataset into training and test folders. I did this to stop noise from dominating the model and to make the classification task cleaner and more reliable.

The core idea was simple: if the visual input contains too many irrelevant features, the CNN or transfer learning model learns the wrong patterns. So I focused on retaining the actual panel region and removing empty or distorted backgrounds. This stage made the data more consistent and improved the quality of the feature maps later in the project.

I also made sure the project followed the expected folder structure, because the notebooks rely on the dataset being organised under Detect_solar_dust/Clean and Detect_solar_dust/Dusty. This requirement was crucial because the pipeline depends on stable paths during image loading and split generation.

Snippet placement guide for Week 6:
- Add Snippet 1 here: this image should show the white-background detection and path preparation code from the feature engineering notebook.
- Suggested placement: immediately after the paragraph about cleaning and filtering images.
- Snippet name: snippets/snippet1.png

[SPACE FOR SNIPPET 1: paste the image from the snippets folder here]

I ended Week 6 with a cleaned and structured dataset, and I had a much clearer understanding of the repository’s direction. At that point, the project had moved from a general idea into an actual image-processing pipeline that was ready for learning experiments.

---

## Week 7: Image augmentation and CNN training workflow

For Week 7, I focused on the next stage of the pipeline: augmentation and the first CNN-based approach. I used the Image Augmentation Notebook to explore how geometric and photometric transformations could make the model more robust. The repository uses Augmentation parameters such as rotation, horizontal and vertical flipping, and brightness adjustments. This was a smart move because dust detection in real environments may vary depending on lighting conditions, angle, or panel orientation.

I specifically used ImageDataGenerator to expand the training set artificially and help the model see variations of the same object. I did this to improve generalisation and reduce overfitting. In a project like solar panel dust detection, a model that only memorises a few perspectives on one brightness setting is not reliable. Data augmentation helps the model learn more stable visual features such as texture differences, panel edges, and dust distribution patterns.

From there, I implemented the CNN feature extractor in the CNN + ML Models notebook. This was where the project began to feel like a true deep-learning workflow. I used a sequential Keras model with multiple Conv2D layers, MaxPooling2D layers, and a Flatten layer before passing the extracted features into classical machine learning classifiers.

The idea was to let the CNN act as a feature extractor and then pass the learned high-level features into models such as Logistic Regression, Decision Trees, SVM, and Random Forest. This hybrid approach was useful because I could combine deep representation learning with classical prediction models. It also made it easier to compare the performance of different classifiers without redesigning the CNN architecture each time.

The model architecture in the repository is simple but effective in a study context: it gradually reduces spatial dimensionality while increasing feature complexity. The network learns edges and textures in early layers and more abstract panel-quality patterns in deeper layers. This is exactly the kind of design that is useful for distinguishing clean solar panels from dusty ones.

Snippet placement guide for Week 7:
- Add Snippet 2 here: this should show the augmentation configuration from the ImageDataGenerator block.
- Add Snippet 3 here: this should show the CNN convolution and feature extraction pipeline.
- Suggested placement: after the explanation of augmentation and before the CNN feature extraction description.
- Snippet names: snippets/snippet2.png and snippets/snippet3.png

[SPACE FOR SNIPPET 2: paste the image from the snippets folder here]

[SPACE FOR SNIPPET 3: paste the image from the snippets folder here]

This stage gave me my first strong insight: the CNN feature extractor was able to convert raw images into meaningful feature vectors, and the classifier stage then determined how well those vectors mapped to the Dusty vs Clean labels. The project now had a clear pipeline that could be evaluated systematically.

---

## Week 8: Transfer learning with Inception, MobileNet, and VGG16

By Week 8, I moved to the transfer learning notebooks, which are the most interesting part of the repository. This stage is where the project becomes more advanced because I no longer started from random weights. Instead, I used pre-trained ImageNet models and adapted them to the dust detection task. I worked with Inception + ML Models, MobileNet + ML Models, and VGG16 + ML Models.

The transfer learning strategy was the same across the notebooks: I loaded a pre-trained model without the top classification layer, froze the base model, and used the output features as the representation for each image. The resulting feature tensor was flattened and then passed into machine learning classifiers. This approach is effective when the dataset is limited or when there is a need to leverage general-purpose image knowledge learned from large-scale datasets.

In the MobileNet notebook, I used the MobileNet base model and then reshaped the predicted feature maps into a 1D vector for downstream classification. The same pattern was applied with VGG16, which has a deeper architecture and more computationally expensive feature maps but often yields high-quality feature representations. The Inception notebook followed the same conceptual logic, with feature extraction driven by an architecture designed to handle different spatial scales efficiently.

This is where I really began comparing model families. Each architecture produced different feature embeddings, and each classifier then mapped those embeddings to the binary outcome. I compared the different pipelines in a practical way: which base model captured dust patterns best, which classifier produced the most stable predictions, and which combination was best for generalisation.

From a development standpoint, this was the most insightful and technically rich phase because it combined deep learning with classical ML evaluation. I was not just training a model; I was benchmarking a system. This is exactly how research-style development projects should move: build the pipeline, test the architecture, and compare the learning representations.

Snippet placement guide for Week 8:
- Add Snippet 4 here: this should show the transfer learning block using MobileNet or VGG16 feature extraction.
- Suggested placement: after the explanation of how the pre-trained model is frozen and flattened before classification.
- Snippet name: snippets/snippet4.png

[SPACE FOR SNIPPET 4: paste the image from the snippets folder here]

I saw that the architecture choice mattered, but so did the downstream classifier. Some feature sets responded better to Logistic Regression, while others suited tree-based or hyperparameter-tuned models. This comparison helped me understand that the strongest system is not always the deepest model; it is the best combination of representation and classifier.

---

## Week 9: Evaluation, deployment, and final project integration

In the final week, I focused on evaluation and deployment. The repository is designed to produce results from a variety of models and then package a trained model into a small web application. The Flask app in cnn_flask_api.py loads a model from the Models folder, accepts an uploaded image, preprocesses it, and returns a clean or dusty prediction. This makes the project feel complete because it transitions from experimentation to a usable inference system.

I also used the README as the project documentation layer. It explains the repository purpose, folder structure, environment setup, and how to run the notebooks. I treated this as the bridge between research work and product-level presentation. Without documentation, the project would be difficult to reproduce, and without deployment logic, it would remain a notebook-only prototype.

From a model perspective, the same logic is repeated across the notebooks: preprocess the images, extract features, train a classifier, evaluate on validation/test data, and save the resulting model. The final deployment step simply takes the trained model, loads it into the Flask server, and exposes a prediction route. This creates an end-to-end workflow that moves from dataset to decision support.

The final application logic is simple but important. It reads the uploaded file, converts the image to RGB, resizes it to 224x224, normalises the pixels, and feeds the array into the loaded model. The model returns a probability score; if the value crosses 0.5, the image is predicted as Dusty, otherwise Clean. This is the same binary classification logic used in the entire training pipeline.

Snippet placement guide for Week 9:
- Add Snippet 5 here: this should show the Flask prediction route and model loading logic.
- Suggested placement: after the paragraph describing deployment and model inference.
- Snippet name: snippets/snippet5.png

[SPACE FOR SNIPPET 5: paste the image from the snippets folder here]

The final outcome of the four weeks was a complete project story: I took a raw dataset, engineered the inputs, trained CNN and transfer-learning pipelines, benchmarked multiple classifiers, and deployed a simple inference API. That is exactly how a development blog should reflect the growth of a machine learning project.

---

## Final reflections

I feel that this project is a strong example of a practical computer-vision workflow. It combines data cleaning, feature engineering, convolutional networks, transfer learning, classical machine learning, and deployment into a single pipeline. The repository’s structure is coherent because each notebook addresses a specific phase of the same problem: understanding the dataset, improving image quality, augmenting the data, building feature extractors, comparing classifiers, and exposing the result in a small Flask app.

The real lesson I learned is that good machine learning systems are not built by just training one model once. They are built through iteration. I improved the data quality first, then compared network architectures, then evaluated different classifiers, and finally packaged the best workflow into a deployment layer. That process is what made the project stronger than a simple notebook demo.

This project blog reflects that journey, and it connects the full repository into a single development narrative.
