# get the data in as usual
#!pip install tf-nightly --quiet
#!pip install python-resize-image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from google.colab import drive
import urllib.request
import os.path
from typing import Optional, List, Callable
from PIL import Image
import shutil, os
import zipfile
import matplotlib.pyplot as plt
from typing import Optional, List, Callable
import numpy as np
import pandas as pd
import shutil, os
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore') # ignore warnings
from imblearn.over_sampling import SMOTE
from keras.models import model_from_json

class Train:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(),
                                      'Colab Notebooks/covid-chestxray-dataset-master/'
                                      'COVID19_images/images_directory')
        self.image_size = (160, 160)
        self.image_shape = self.image_size + (3,)
        self.batch_size = 32
        self.test_loss = 999999
        self.acc = 999999
        self.model = 0

    def get_data(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(self.data_path,
                                                                       validation_split=0.25,
                                                                       subset='training',
                                                                       seed=1337,
                                                                       image_size=self.image_size,
                                                                       batch_size=self.batch_size)
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(self.data_path,
                                                                      validation_split=0.25,
                                                                      subset='validation',
                                                                      seed=1337,
                                                                      image_size=self.image_size,
                                                                      batch_size=self.batch_size)
        self.train_dsu = tf.data.Dataset.unbatch(train_ds)
        self.test_dsu = tf.data.Dataset.unbatch(test_ds)
        print ("done getting data")

    def training(self, train_dsu, modelname="my_model_name", num_folds=3, num_classes=3):
        # K-fold Cross Validation model evaluation
        acc_per_fold = []
        loss_per_fold = []
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        input_shape = self.image_shape
        no_classes = 3
        x = get_k_splits(train_dsu)  # split into training and testing set

        fold_no = 0
        for train, val in kfold.split(x[0], x[1]):
            unique, counts = np.unique(x[1][train], return_counts=True)
            # SMOTE
            sm = SMOTE(random_state=42)
            hold_shape = x[0][train].shape
            X_train = x[0][train].reshape(hold_shape[0], hold_shape[1] * hold_shape[2] * hold_shape[3])
            y_train = x[1][train].reshape(hold_shape[0], 1)
            X_smote, y_smote = sm.fit_resample(X_train, y_train)
            new_shape = X_smote.shape[0]
            a = X_smote.reshape(new_shape, hold_shape[1], hold_shape[2], hold_shape[3])
            b = y_smote.reshape(new_shape, )
            X_smote = a
            y_smote = b
            del a
            del b
            del X_train
            del y_train
            unique, counts = np.unique(y_smote, return_counts=True)
            # batch it up before running model
            train_generator = batch_generator(X_smote, y_smote, self.batch_size)
            del X_smote
            del y_smote

            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
            base_model = tf.keras.applications.MobileNetV2(input_shape=self.image_shape,
                                                           include_top=False,
                                                           weights='imagenet')

            model = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip('horizontal'),
                                         layers.experimental.preprocessing.RandomRotation(0.1),
                                         base_model,
                                         tf.keras.layers.GlobalAveragePooling2D(),
                                         tf.keras.layers.Dense(3, activation='softmax')])

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),  # smaller learning rate
                          loss="SparseCategoricalCrossentropy",  # weighted_ce ,
                          metrics=['accuracy'])

            history = model.fit(train_generator,  # X_smote, y_smote,
                                batch_size=self.batch_size,
                                validation_data=(x[0][val], x[1][val]),
                                epochs= 100, # 100,
                                steps_per_epoch= 100, #100,
                                class_weight={0: 4., 1: 26., 2: 16.})
            self.model = model
            self.save_model(modelname) # save the model

            # Generate generalization metrics
            scores = model.evaluate(x[0][val], x[1][val], verbose=0)
            # print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]};
            # {model.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            predictions = model.predict(x[0][val])
            labels = np.argmax(predictions, axis=-1)
            target_names = ['class 0', 'class 1', 'class 2']
            my_eval = classification_report(y_true=x[1][val],
                                            y_pred=labels,
                                            target_names=target_names,
                                            output_dict=True)
            my_eval_df = pd.DataFrame(my_eval).transpose()
            pairs = [(i, j) for i, j in zip(labels, x[1][val])]
            print(Counter(elem for elem in pairs))
            temp = dict(Counter(elem for elem in pairs))

            for j in range(num_classes):
                name = "pred_" + str(j)
                my_eval_df[name] = [0, 0, 0, "-", "-", "-"]
                for i in range(num_classes):
                    if (j, i) in temp.keys():
                        my_eval_df[name][i] = temp[(j, i)]
            my_eval_df.to_csv(modelname + '_classification_report_' + str(fold_no) + '.csv', index=True)

            #  "Accuracy"
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(modelname + "_accuracy_" + str(fold_no) + ".png")
            plt.show()

            # "Loss"
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(modelname + "_loss_" + str(fold_no) + ".png")
            plt.show()
            fold_no = fold_no + 1

        # Save results
        temp = pd.DataFrame(list(zip(acc_per_fold, loss_per_fold)),
                            columns=['Accuracy', 'Loss'])
        temp.loc['mean'] = temp.mean()
        print(temp)
        temp.to_csv(modelname + '.csv', index=True)

    def save_model(self, modelname):
            model_json = self.model.to_json()
            with open("model_" + modelname + time.strftime("%Y%m%d-%H%M%S") + ".json", "w") as json_file:
                json_file.write(model_json)
            self.model.save_weights("model_" + modelname + time.strftime("%Y%m%d-%H%M%S") + ".h5")
            print(time.strftime("%Y%m%d-%H%M%S") + "Saved model to disk \n")

    def evaluate_on_test(self, modelname="my_model_name", num_classes=3):
        y = get_k_splits(self.test_dsu)
        self.test_loss, self.test_acc = self.model.evaluate(y[0], y[1], verbose=0)
        # test data
        test_predictions = self.model.predict(y[0])
        test_labels = np.argmax(test_predictions, axis=-1)
        target_names = ['class 0', 'class 1', 'class 2']
        test_my_eval = classification_report(y_true=y[1],
                                             y_pred=test_labels,
                                             target_names=target_names,
                                             output_dict=True)
        test_my_eval_df = pd.DataFrame(test_my_eval).transpose()
        test_pairs = [(i, j) for i, j in zip(test_labels, y[1])]
        test_temp = dict(Counter(elem for elem in test_pairs))
        for j in range(num_classes):
            name = "pred_" + str(j)
            test_my_eval_df[name] = [0, 0, 0, "-", "-", "-"]
            for i in range(num_classes):
                if (j, i) in test_temp.keys():
                    test_my_eval_df[name][i] = test_temp[(j, i)]
        print(test_my_eval_df)
        test_my_eval_df.to_csv(modelname + '_test_classification_report_' + '.csv', index=True)


def get_k_splits(data_dsu): # get to the data form that we want
    all_images = []
    all_labels = []
    for image, label in data_dsu.take(-1):
        all_images.append(image.numpy())
        all_labels.append(int(label))
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    print("done get_k_splits")
    return (all_images, all_labels)


def batch_generator(X, Y, batch_size=32):
    indices = np.arange(len(X))
    batch = []
    while True:
        # shuffle your data before each epoch
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                yield X[batch], Y[batch]
                batch = []

if __name__ == "__main__":
    my_train = Train()
    my_train.get_data()
    my_train.training(my_train.train_dsu, "temp_model")

