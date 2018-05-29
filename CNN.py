import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Convolution1D, Embedding, MaxPooling1D, Flatten
from keras.layers import GlobalMaxPooling1D
import preprocessing
import os
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))



embedding_dim = 128
# Training parameters
batch_size = 64
num_epochs = 20
sequence_length = 79



print("Data Loading...")

x,y, vocabulary, vocabulary_inv_list = preprocessing.load_data()


vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

# Shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]
train_len = int(len(x) * 0.8)
x_train = x[:train_len]
y_train = y[:train_len]
x_test = x[train_len:]
y_test = y[train_len:]

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_test)
print('vocabulary_inv len' + str(len(vocabulary_inv)))
print("x_train static shape:", x_train.shape)
print("x_test static shape:", x_test.shape)
print("- x_train -")
print(x_train)
print()
print('- x_test -')
print(x_test)
print()
print(x_train)

input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

z = Embedding(len(vocabulary_inv),embedding_dim,input_length=sequence_length,name='embedding_layer')(model_input)
z = Dropout(0.3)(z)
conv = Convolution1D(filters=128,
                     kernel_size=2,
                     padding='valid',
                     activation='relu',
                     strides=1)(z)
conv = GlobalMaxPooling1D()(conv)
z = Dense(128,activation='relu')(conv)
z = Dropout(0.3)(z)
model_output = Dense(2,activation='softmax')(z)
model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())



custom_hist = CustomHistory()
custom_hist.init()


model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1,verbose=1,callbacks=[custom_hist])



training_loss, training_accuracy = model.evaluate(x_train, y_train,verbose=1)
test_loss, test_accuracy = model.evaluate(x_test, y_test,verbose=1)



y_prob = model.predict(x_test)
y_classes = y_prob.argmax(axis=-1)


print("Test Y : ",y_test.argmax(axis=-1))
print("Predict Y : ", y_classes)


print("--- Training Result ---")
print("Accuracy : " + str(round(training_accuracy,2)))
print("-----------------------")
print()

print("--- Test Result ---")
print("Accuracy : " + str(round(test_accuracy,2)))
print("-----------------------")
model.save('model.h5')

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
modelfile = 'model.png'
plot_model(model, to_file=modelfile, show_shapes=True, show_layer_names=True)



# %matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(custom_hist.train_loss, 'y', label='train loss')
loss_ax.plot(custom_hist.val_loss, 'r', label='val loss')

acc_ax.plot(custom_hist.train_acc, 'b', label='train acc')
acc_ax.plot(custom_hist.val_acc, 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()



