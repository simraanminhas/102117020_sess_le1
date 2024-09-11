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

## Next Steps
To complete this process, the following steps are recommended:

Access to GPU: Running the training and fine-tuning process on a GPU to significantly speed up convergence and allow for completion of the fine-tuning.
Model Evaluation: Once training is completed, evaluate the model's accuracy, loss, and performance on the test set.
Further Optimization: Fine-tune hyperparameters such as learning rate and batch size for better performance and potentially introduce early stopping to prevent overfitting.

This project successfully prepared a custom dataset of 30 recordings for each of the 35 speech commands and set up a CNN classifier for fine-tuning. However, due to hardware constraints, the fine

-----------------------------


# CODE SNIPPETS

![image](https://github.com/user-attachments/assets/c924c4c4-abea-4b33-85a3-b4fe43340ee7)
![image](https://github.com/user-attachments/assets/2aac81d7-6439-4c32-8d6f-a5da20ea1abe)
![image](https://github.com/user-attachments/assets/b7f998a1-a125-4fe2-bba9-87432f7f7100)
![image](https://github.com/user-attachments/assets/7a4499bf-1840-473d-aff0-9ad6fd8bd302)

### Drive link to self created dataset and the Report on Research paper is attached in the repository itself.



