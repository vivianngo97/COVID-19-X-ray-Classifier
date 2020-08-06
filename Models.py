# This is code to run the models

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import time
from keras.models import model_from_json
from collections import Counter
from sklearn.metrics import classification_report

def load_model(json_name, h5_name):
    """
    :param json_name: Name of the json file
    :type json_name: string
    :param h5_name: Name of the h5 file holding model weights
    :type h5_name: string
    :return: loaded_model
    :rtype: tensorflow model
    """
    print(time.strftime("%Y%m%d-%H%M%S") + "loading model")
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_name)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                         loss= "SparseCategoricalCrossentropy",
                         metrics=['accuracy'])
    return loaded_model


def evaluations_table(model, pred_x=[], true_y=[]):
    """
    :param model: The model that we are using to make predictions
    :type model: tensorflow model
    :param pred_x: input images
    :type pred_x: numpy array
    :param true_y: true classes
    :type true_y: numpy array
    :return:
    :rtype:
    """
    test_predictions = model.predict(pred_x)
    test_labels = np.argmax(test_predictions, axis=-1)
    target_names = ['class 0', 'class 1', 'class 2']
    test_my_eval = classification_report(y_true=true_y, y_pred=test_labels, target_names=target_names, output_dict=True)
    test_my_eval_df = pd.DataFrame(test_my_eval).transpose()
    test_pairs = [(i, j) for i, j in zip(test_labels, true_y)]
    print(Counter(elem for elem in test_pairs))
    test_temp = dict(Counter(elem for elem in test_pairs))
    for j in range(3): # because we have specifically three classes
        name = "pred_" + str(j)
        test_my_eval_df[name]=[0, 0, 0, "-", "-", "-"]
        for i in range(3):
            if (j, i) in test_temp.keys():
                test_my_eval_df[name][i] = test_temp[(j, i)]
    return test_my_eval_df # classification report AND contingency table


def predict_one_image(model, image, labels_string=["bacterial","covid","healthy"]):
    img = mpimg.imread(image) #
    # example:'images_directory/covid/gr2_lrg-c.png')
    # check if the image is grayscale or not
    imgplot = plt.imshow(img) # plot the image
    if len(img.shape) == 2:
        # i.e. grayscale
        img = tf.expand_dims(img, 0)
        rgb_batch = np.repeat(img[..., np.newaxis], 3, -1)
        rgb_batch = rgb_batch.astype(np.float32)
        rgb_batch = tf.image.resize(rgb_batch, size=(160, 160))  # resizing image
        Prob = model.predict(rgb_batch)  # prediction
        indd = tf.argmax(Prob[0], axis=-1).numpy()
        return labels_string[indd]
    elif len(img.shape)==3:
        # not grayscale
        img = tf.expand_dims(img, 0)
        img = tf.image.resize(img, size=(160, 160))  # resizing image
        Prob = model.predict(img)  # prediction
        indd = tf.argmax(Prob[0], axis=-1).numpy()
        return labels_string[indd]
    else:
        return "Error: wrong image shape"



# converts to numpy dataframe so we can just get the images or the labels
def get_k_splits(train_dsu):
# converts to numpy dataframe s
    all_images = []
    all_labels = []
    for image, label in train_dsu.take(-1):
        all_images.append(image.numpy())
        all_labels.append(int(label))
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    print("done")
    return (all_images, all_labels)  # , skf, fold_inds)
