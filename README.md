# COVID and Bacterial Pneumonia Detector
# Table of Contents  
- [Overview](#Overview) 
- [Data Preprocessing](#Data-Preprocessing)  
- [Model](#Model)  
- [Evaluation](#Evaluation)  
- [Future Considerations](#Future-Considerations)
- [Examples](#Examples)
- [Test it out!](#Test-it-out)

# Overview

As of July 2020, the global pandemic COVID-19 has infected over 15 million people and caused major social and economic change. Due to the novelty and rapid spread of the virus, the healthcare system has experienced significant strain in its ability to diagnose, treat, and manage patients who become infected. Currently, there is a limited capacity for diagnostic testing in Canada. Tests are recommended to individuals who are at high risk or may have been in contact with someone else with the infection. Given the importance of early detection for mitigating the spread of COVID-19, we propose alternative methods of diagnostic testing. Pneumonia may be present in severe cases of COVID-19, and can be identified by radiologists on x-rays. This project focuses on the classification of bacterial pneumonia, COVID-19, and healthy x-ray images using convolutional neural networks. A strong classification method will help to reduce the need for COVID-19 specific testing kits and reduce the strain on healthcare workers.

In this repo, we train a deep learning model to distinguish between x-ray images of healthy individuals and those with COVID-19 or bacterial pneumonia. To read our blog regarding this project, please visit [https://covidpneumoniaclassifier.wordpress.com/](https://covidpneumoniaclassifier.wordpress.com/). 

# Data Processing 

The data used throughout this project was collected from several sources. The first data source is ~\cite{cohen2020covid}, from which we obtained x-rays and CT scans of roughly 200 patients with COVID-19. This dataset is updated regularly with the approval of the University of Montreal's Ethics Committee.

We also required medical images of healthy individuals and individuals with bacterial pneumonia. For this, we utilized a dataset from Kaggle of chest x-ray images for individuals with and without bacterial pneumonia ~\cite{pneumoniadata}. This dataset contains more than 5,000 x-ray images all together.


# Model 

# Evaluation 

# Future Considerations 

# Examples 

# Test it out

To test out this model for yourself, please visit our small user-friendly Github repo at [https://github.com/vivianngo97/COVID-Bacterial-Pneumonia-Classifier-Run-Model](https://github.com/vivianngo97/COVID-Bacterial-Pneumonia-Classifier-Run-Model). There will be instructions on how to run the model and classify your own x-ray images.
