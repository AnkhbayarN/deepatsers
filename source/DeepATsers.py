import torch
from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from imblearn.metrics import specificity_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def _importData(file_name, classification, columns):
    ds = pd.read_csv(file_name, low_memory=False)
    y = ds[classification]
    x = ds.drop(columns, axis = 1)
    return x, y

# Define the model
def _createModel(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Input shape: (sequence_length, input_dim)
        tf.keras.layers.Conv1D(filters = 16, kernel_size = 8, strides = 1, activation=None),  # 1st convolutional layer: 32 filters, size 20x1
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),  # Add ReLU activation after Batch Normalization
        tf.keras.layers.Conv1D(filters = 32, kernel_size = 8, strides = 1, activation=None),  # 2nd convolutional layer: 64 filters, size 20x1
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('tanh'),  # Add ReLU activation after Batch Normalization
        # Add more layers as needed
        tf.keras.layers.Flatten(),  # Flatten layer to convert 3D output to 2D for dense layers
        tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(8, activation='tanh'),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
        
    return model

def _fit(model, x_train, y_train, x_test, y_test, batch, epoch):
    history = model.fit(x_train, y_train, batch_size = batch, epochs = epoch, validation_data = (x_test, y_test))
    # history = model.fit(x_train, y_train, batch_size = batch, epochs = epoch)
    return history

def _predict(model, x_test):
    pred = model.predict(x_test)
    y_pred = pred.argmax(axis=-1)
    return y_pred, pred

def _conf_matrix(labels, title, y_test, y_pred, multi_or_binary, epsname):
    cm = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_title(title)
    plt.rcParams.update({'font.size':20})
    if (multi_or_binary == 'binary'):
        cmd.plot(ax=ax, cmap=plt.cm.Blues, values_format='g')
        ax.tick_params(axis='y', rotation=90)
        for label in ax.get_yticklabels():
            label.set_verticalalignment('center')

    else:
        cmd.plot(xticks_rotation=90, ax=ax, cmap=plt.cm.Blues, values_format='g')
            
def _report(labels, y_test, y_pred):
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

def _specificity(y_test, y_pred):
    print('specificity:', specificity_score(y_test, y_pred, average='weighted'))

def _auc(y_test,y_pred_proba):
    # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    return auc

def _accuracy_by_model(history):
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def _model_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


input_shape = (1331, 1)  # Input shape for 1D spectra data
num_classes = 5  # Number of output classes

labels = ['Blank', 'Streptavidin', 'VLP', 'S', 'N']
# labels = ['Negative', 'Positive']
# columns = ['protein', 'diagnosis', 'multiclassification', 'folder', 'subfolder',	'binaryclassification',	'class', 'concentration']
columns = ['multiclassification']

file_train = 'cv_folds_multi_augmented/train_fold_1.csv'
file_test = 'cv_folds_multi_augmented/test_fold_1.csv'
classification = 'multiclassification'
title = 'DeepATsers'


x, y = _importData(file_train, classification, columns)
x_test, y_test = _importData(file_test, classification, columns)

model = _createModel(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = _fit(model, x, y, x_test, y_test, 32, 10)
y_pred, y_pred_proba = _predict(model, x_test)

epsname = 'cnn_augmented_allpeak_pc'
labels = ['Blank signal', 'Streptavidin', 'VLP', 'S protein', 'N protien']
_report(labels, y_test, y_pred)
_conf_matrix(labels, title, y_test, y_pred, 'multi', epsname)
_specificity(y_test, y_pred)
_auc(y_test, y_pred_proba)

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

# Binarize the output
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)

# Initialize variables for storing ROC and AUC values
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Plotting the ROC curve for each class
plt.figure(figsize=(6, 5))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
            # label=f'ROC curve for {labels[i]} (area = {roc_auc[i]:.2f})')
            label=f'{labels[i]} AUC={roc_auc[i]:.4f}')  # Change the label format here

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.grid(True)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.tick_params(axis='both', labelsize=12)  # Adjust label size for both axes
plt.savefig('diagram/roc.eps', bbox_inches='tight')
plt.show()