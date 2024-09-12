# 102117020_sess_le1

## Report on Fine-Tuning CNN Model for Speech Command Recognition

## Introduction
The purpose of this project was to fine-tune a Convolutional Neural Network (CNN) model for speech command recognition using a custom dataset. The dataset consisted of 30 recordings for each of the 35 possible words, all recorded in my own voice. This project follows the structure of the Speech Commands dataset commonly used for keyword spotting tasks. The aim was to adapt the existing model to recognize my voice and fine-tune the classifier accordingly.

## Dataset Creation
For this project, I recorded 30 short audio samples for each of the 35 speech command words, resulting in a custom dataset. Each audio sample was recorded using a consistent sampling rate of 16 kHz to match typical input requirements for speech command models. The words recorded included simple commands such as "yes", "no", "up", "down", "left", "right", among others.

Once the recordings were completed, the dataset was compressed into a ZIP file and mounted in the environment. The following steps were followed for dataset extraction:

Uploading: The dataset was uploaded in ZIP format to the working directory.
Extraction: The ZIP file was extracted into a folder named dataset, allowing the audio files to be used for fine-tuning the CNN model.
## Model Overview
We used the M5 CNN architecture, which is well-suited for processing one-dimensional time-series data like audio. The key layers of the model included:

Convolutional layers: To extract features from the waveform.
Batch normalization: To stabilize learning and speed up convergence.
Max-pooling layers: For dimensionality reduction.
Fully connected layer: For classification across 35 output classes (the speech commands).
The model was trained using the following parameters:

Loss function: CrossEntropyLoss.
Optimizer: Adam optimizer with a learning rate of 0.001.
Scheduler: Learning rate scheduler to reduce learning after 20 epochs.
Fine-Tuning Process
After extracting the dataset, we proceeded to load it into a PyTorch DataLoader for use in the fine-tuning process. The dataset was split into training (80%) and testing (20%) sets, ensuring that each set had a representative sample of the recorded commands.

The training loop involved iterating through the dataset, applying transformations to convert the waveform data into mel-spectrograms, and updating the model’s weights through backpropagation. The testing loop evaluated the model’s accuracy after each epoch.

## Challenges
Unfortunately, due to the limitations of the available hardware, specifically the lack of access to a GPU, the fine-tuning process was not fully completed. The model's training was started, but the process was too slow to achieve meaningful results within the available time. Consequently, the accuracy and loss of the model were not optimized, and the model could not be fully evaluated on the test set.

## Recording Audio Samples:
Recorded 30 audio samples, each containing 35 distinct words. Used a laptop voice recorder for capturing the audio.

## Dataset Preparation:
Compiled the recorded audio samples adding them to google drive. Mounted the drive to the working environment. Extracted the zip file to access the individual audio files for processing.

## Data Extraction:
Organized and processed the audio files for model training. Converted audio files to mono if necessary and resampled to 8000 Hz.

## Custom Dataset Class
Created a custom dataset class MySpeechCommands to load and preprocess the audio data. Implemented methods to convert stereo audio to mono, resample audio, and map labels to indices

## CNN Model Classifier Development:
Developed and configured a Convolutional Neural Network (CNN) model named M5 for keyword spotting. The model includes several 1D convolutional layers, batch normalization, pooling layers, and a fully connected layer for classification.

## Dataset Splitting:
Prepared the dataset for training, validation, and testing. Split the dataset into training, validation, and testing subsets using custom file lists.

## Fine-Tuning Process:
Fine-tuned the CNN model using the prepared dataset. Achieved a validation accuracy of 0.8190 after 10 epochs of training.

## Evaluation:
Evaluated the model on the test set to assess its performance on unseen data. The model’s performance metrics include an accuracy of 0.0286 on the test set (initial run), indicating further tuning and debugging needed.
-----------------------------


# CODE SNIPPETS

![image](https://github.com/user-attachments/assets/c924c4c4-abea-4b33-85a3-b4fe43340ee7)
![image](https://github.com/user-attachments/assets/2aac81d7-6439-4c32-8d6f-a5da20ea1abe)
![image](https://github.com/user-attachments/assets/b7f998a1-a125-4fe2-bba9-87432f7f7100)
![image](https://github.com/user-attachments/assets/7a4499bf-1840-473d-aff0-9ad6fd8bd302)

### Drive link to self created dataset and the Report on Research paper is attached in the repository itself.
### COLAB NOTEBOOK IS IN THE REPO ITSELF.


