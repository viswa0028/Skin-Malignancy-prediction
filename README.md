# Skin Malignancy Detection Model

A deep learning model for detecting malignant skin lesions using convolutional neural networks. This project aims to assist dermatologists in identifying skin Malignancy.
## Overview 
The initial CNN models/ Vision Transformers just take the input images and train/fine-tune their architecture and predict the output and the accuracy for those models might not be as good as desired.
So we tackled this problem by introducing a feature fusion architecture where the model not only takes the images as the input but also captures various information from the images using Computer Vision and adds them to the training field which will further increase the knowledge background to the models.

## Dataset

The dataset used was ISIC 2018 Training dataset for training. The ISIC 2018 dataset is a prominent, public repository of over 10,000 annotated dermoscopic skin lesion images released by the International Skin Imaging Collaboration (ISIC). It is widely used in medical AI research to train models for automated skin cancer diagnosis, lesion segmentation, and attribute detection, featuring seven diagnostic classes including melanoma and benign nevus. 

## Features Extraction

The feature extraction from the images are like mean,standard deviation and so on using computer Vision and integrating them to the dataset original dataset and fine tuning the vision transformers to increase the accuracy prediction of the models.

## Tech Stack

The Tech Stack used for developing this pipeline was:
 - torch
 - scikit-learn
 - Pillow
 - numpy<2
 - opencv-python
 - pandas

## Conclusion
By adding these additional features and training the Vision Transformers we have seen an increase the accuracy by over 8-10% and the final overall accuracy for some models are reaching over 94% which is a very good performance for prediction of the skin cancer and could be used in real time.

## Future Directions

 -  The models could be further introduced with multiple more features to increase the accuracy further than now.
 -  Deploying the model and using them in real time.
