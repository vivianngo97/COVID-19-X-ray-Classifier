# COVID and Bacterial Pneumonia Detector
# Table of Contents  
- [Overview](#Overview) 
- [Data Preprocessing](#Data-Preprocessing)  
- [Model](#Model)  
- [Evaluation](#Evaluation)  
- [Test it out!](#Test-it-out)

# Overview

As of July 2020, the global pandemic COVID-19 has infected over 15 million people and caused major social and economic change. Due to the novelty and rapid spread of the virus, the healthcare system has experienced significant strain in its ability to diagnose, treat, and manage patients who become infected. Currently, there is a limited capacity for diagnostic testing in Canada. Tests are recommended to individuals who are at high risk or may have been in contact with someone else with the infection. Given the importance of early detection for mitigating the spread of COVID-19, we propose alternative methods of diagnostic testing. Pneumonia may be present in severe cases of COVID-19, and can be identified by radiologists on x-rays. This project focuses on the classification of bacterial pneumonia, COVID-19, and healthy x-ray images using convolutional neural networks. A strong classification method will help to reduce the need for COVID-19 specific testing kits and reduce the strain on healthcare workers.

In this repo, we train a deep learning model to distinguish between x-ray images of healthy individuals and those with COVID-19 or bacterial pneumonia. To read our blog regarding this project, please visit [https://covidpneumoniaclassifier.wordpress.com/](https://covidpneumoniaclassifier.wordpress.com/). 

![bacterial_xray](https://github.com/vivianngo97/COVID-19-X-ray-Classifier/blob/master/fixtures/bac1.jpeg)
![covid_xray](https://github.com/vivianngo97/COVID-19-X-ray-Classifier/blob/master/fixtures/covid1.jpg)
![healthy_xray](https://github.com/vivianngo97/COVID-19-X-ray-Classifier/blob/master/fixtures/NORMAL2-IM-1440-0001.jpeg)


# Data  

The data used throughout this project was collected from several sources. The first data source is [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset), from which we obtained x-rays and CT scans of roughly 200 patients with COVID-19. This dataset is updated regularly with the approval of the University of Montreal's Ethics Committee.

We also required medical images of healthy individuals and individuals with bacterial pneumonia. For this, we utilized a dataset from Kaggle of chest x-ray images for individuals with and without bacterial pneumonia. This dataset contains more than 5,000 x-ray images all together and can be accessed at [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

For the purposes of this project, we used 4,754 images (2,780 bacterial pneumonia, 391 COVID-19, and 1,583 healthy images). This data was split into a 75% training set and 25% testing set.  

# Model 

Because the dataaset is small, we leveraged transfer learning to improve model performance. This model uses MobileNetV2. We improved on the pre-trained MobileNetV2 model by adding our own global pooling and prediction layer, as well as experimenting with the hyperparameters to determine the optimal network configuration. 

The final model that we selected is called P6SMOTE (please see our paper for more information) because of its high macro average F1-scores as well as its stable increasing accuracy. This model was trained with a learning rate of 0.0001, 100 epochs, 100 steps per epoch, SMOTE, and class weights of {0: 4, 1: 26, 2: 12}. 

# Evaluation 

We used the final model to make predictions on the reserved test set, which comprised 25% of our total dataset. The x-rays in the test set were not used in any form during the training and validation process. The precision, recall, F1-score, accuracy and confusion matrix of our model is displayed in the table below. pred_i indicates images that were predicted into class i.

The results are positive, with a macro average precision, recall, and F1-score of 0.972, 0.959, and 0.965 respectively. The confusion matrix also shows that no classes are ignored when making predictions. This informs us that our model appears to be generalizable to new unseen data. Our final accuracy of 96.8\% also outperforms other recent studies that use x-ray imaging to detect COVID-19. 


|              | precision | recall | f1-score | support | pred_0 | pred_1 | pred_2 |
| ------------ | --------- | ------ | -------- | ------- | ------ | ------ | ------ |
| class 0      | 0.975     | 0.971  | 0.973    | 687     | 667    | 1      | 19     |
| class 1      | 0.989     | 0.937  | 0.962    | 95      | 5      | 89     | 1      |
| class 2      | 0.952     | 0.970  | 0.961    | 406     | 12     | 0      | 394    |
| accuracy     | 0.968     | 0.968  | 0.968    |         | -      | -      | -      |
| macro avg    | 0.972     | 0.959  | 0.965    | 1188    | -      | -      | -      |
| weighted avg | 0.968     | 0.968  | 0.968    | 1188    | -      | -      | -      |


# Test it out

To test out this model for yourself, please visit our small user-friendly Github repo at [https://github.com/vivianngo97/COVID-Bacterial-Pneumonia-Classifier-Run-Model](https://github.com/vivianngo97/COVID-Bacterial-Pneumonia-Classifier-Run-Model). There will be instructions on how to run the model and classify your own x-ray images.
