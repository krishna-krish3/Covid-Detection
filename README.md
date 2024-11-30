# COVID-19 Detection Using X-Ray Images
This project focuses on detecting COVID-19 from X-ray images using a deep learning approach. The model leverages PyTorch and DenseNet architecture to classify images with high accuracy.

## Features
Preprocessing of X-ray datasets with resizing, normalization, and augmentation.
Implementation of training, validation, and testing pipelines.
Utilization of DenseNet for feature extraction and classification.
Performance optimization using GPU on Google Colab.
Visualization of training/validation loss and accuracy over epochs.

## Dataset
The dataset consists of 7,330 labeled X-ray images. The data is split into:

Training Set: 80% \
Validation Set: 10% \
Testing Set: 10%

## Requirements
To run this project, you need the following libraries and tools:

Python 3.8+ \
PyTorch \
torchvision \
numpy \
matplotlib \
Google Colab (optional for GPU) 

## **Install dependencies using pip:**
```bash
pip install torch torchvision matplotlib
```

## Model Architecture:
The project uses a DenseNet architecture pretrained on ImageNet, with the final fully connected layer fine-tuned for binary classification (COVID-19 positive/negative).

## How to Run
Clone the repository:
``` bash
git clone https://github.com/yourusername/covid-detection.git
cd covid-detection
```

Upload the dataset to the Covid_Images directory.

Run the training script:
python train.py

Evaluate the model on the test set:
python test.py

Results
Training and validation loss/accuracy curves are visualized for performance tracking.
The model achieves high accuracy on the test set (update with specific metrics after evaluation).

Acknowledgments
The project uses pretrained DenseNet models from torchvision and data augmentation techniques for better generalization.
Thanks to publicly available X-ray datasets for enabling this research.

Future Work
Extend the model to classify multiple lung conditions.
Improve accuracy with advanced architectures like Vision Transformers.
