
from scipy import rand
from sklearn.model_selection import learning_curve
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import keras
from keras import metrics

w,h = 48, 48
X_data = np.load("faces.npy")
Y_data = np.load("ages.npy")
X_data = X_data.astype('float32') / 255
X_data = X_data.reshape(X_data.shape[0], w, h, 1)
X_data = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_data))

@tf.function
def shuffle(X_data, Y_data):
    seeed = np.random.randint(1,10)
    x = tf.random.shuffle(X_data, seed= seeed)
    y = tf.random.shuffle(Y_data, seed= seeed)
    return x, y
X_data, Y_data = shuffle(X_data, Y_data)

(x_tr_va, x_test) = X_data[int(X_data.shape[0]*0.2):], X_data[:int(X_data.shape[0]*0.2)] 
(y_tr_va, y_test) = Y_data[int(Y_data.shape[0]*0.2):], Y_data[:int(Y_data.shape[0]*0.2)]

(x_train, x_val) = x_tr_va[int(x_tr_va.shape[0]*0.1):], x_tr_va[:int(x_tr_va.shape[0]*0.1)] 
(y_train, y_val) = y_tr_va[int(y_tr_va.shape[0]*0.1):], y_tr_va[:int(y_tr_va.shape[0]*0.1)]

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
# 1st Convolution Block
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(48,48,3)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))

# 2nd Convolution Block
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))

# 3rd Convolution Block
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))

# 4th Convolution Block
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))
X_data, Y_data = shuffle(X_data, Y_data)

# 5th Convolution Block
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dense(1))


# Take a look at the model summary
model.summary()
# For Grid Search of Hyperparameters
Epoch=[5, 10, 15, 30]
Mb=[16, 32, 64, 128]
# Epoch=[30]
# Mb=[64]

best_HP=np.array(2)
min_loss=np.Inf
print("\n----------------------------------------------------------------------")
print("Performing Grid Search for Combinations of Hyperparameters:")
print("----------------------------------------------------------------------")

for epoch in Epoch:
    for mb in Mb:
        HP= [epoch, mb]
        print("\nTraining on Hyperparameters:",HP)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.RootMeanSquaredError()])
        checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
        result_history=model.fit(x_train,
                y_train,
                batch_size=mb,
                epochs=epoch,
                validation_data=(x_val, y_val),
                callbacks=[checkpointer])
        va_loss= result_history.history['val_loss'][0]
        if(va_loss<min_loss):
            min_loss=va_loss
            best_HP= HP
            print("Hyperparameters Updated")
        else:
            print("Hyperparameters Not Updated")
        print("Best Hyperparameter uptill now:",best_HP)

epoch, mb = best_HP

print("\n----------------------------------------------------------------------")
print(" Grid Search Completed")
print("----------------------------------------------------------------------")
print(" Results after performing Grid Search:")
print(" Best Hyperparameters:")
print("   Epochs= ",epoch)
print("   Mini Batch Size= ", mb)

print("\n----------------------------------------------------------------------")
print("ReTraining on Best Hyperparameters :")
print("----------------------------------------------------------------------")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.RootMeanSquaredError()])

checkpointer = ModelCheckpoint(filepath='model1.weights.best.hdf5', verbose = 1, save_best_only=True)
model.fit(x_train,
        y_train,
        batch_size=mb,
        epochs=epoch,
        validation_data=(x_val, y_val),
        callbacks=[checkpointer])

print("\n----------------------------------------------------------------------")
print("Performance Evaluation")
print("----------------------------------------------------------------------")

# Load the weights with the best validation accuracy
model.load_weights('model1.weights.best.hdf5')

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print RMSE loss
print('\n', 'RMSE Loss on Test Dataset:', score[1])
