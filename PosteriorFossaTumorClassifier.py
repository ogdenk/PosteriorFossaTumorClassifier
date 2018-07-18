import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import models
from keras import layers
import matplotlib.pyplot as plt

trainFrac = 0.8 #Training fraction (includes validation)
valFrac = 0.2 #Fraction of training data used for validation

rootDir = 'd:/'
filename = 'rawavg.csv'
datafile = os.path.join(rootDir, filename)

data = pd.read_csv(datafile,index_col='Pat')
data = shuffle(data)

trainRows = int(trainFrac*data.shape[0])

train_data_df, test_data_df = np.split(data,[trainRows])

train_data_size = len(train_data_df)
test_data_size = len(test_data_df)

train_labels = train_data_df['SzBd400'].as_matrix().astype('float32')
test_labels = test_data_df['SzBd400'].as_matrix().astype('float32')

train_data = train_data_df.as_matrix()[:,1:]
test_data = test_data_df.as_matrix()[:,1:]

means = train_data.mean(axis=0)
sigmas = train_data.std(axis=0)

train_data = (train_data-means)/sigmas
test_data = (test_data-means)/sigmas

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_data,train_labels, epochs=30, batch_size=64, validation_data=(test_data, test_labels))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
