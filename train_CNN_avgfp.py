import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
import tensorflow as tf
import pickle
import gc
from sklearn.model_selection import train_test_split

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

avgfp = pd.read_csv("avgfp.tsv", sep = "\t")
y = avgfp["score_wt_norm"].to_list()
y = y[:-2001]

enc_avgfp = np.load("avgfp.npy")
enc_avgfp = enc_avgfp[:-2001]
enc_seqs = np.expand_dims(enc_avgfp,1)

X_train, X_test, y_train, y_test = train_test_split(enc_seqs,y,  test_size=0.2)

def build_model(model_type='regression', conv_layer_sizes=(16, 16, 16), dense_layer_size=16, dropout_rate=0.25):
    """
    """
    # make sure requested model type is valid
    if model_type not in ['regression', 'classification']:
        print('Requested model type {0} is invalid'.format(model_type))
        sys.exit(1)
    #nb_classes = 2
    nb_kernels = 3
    nb_pools = 2
    batch_size = 10
    n_epochs = 80
    
    model = models.Sequential()

    model.add(layers.ZeroPadding2D((1,1), input_shape = (1,237, 40)))
    model.add(layers.Conv2D(conv_layer_sizes[0], (nb_kernels, nb_kernels), activation='relu'))
    model.add(layers.MaxPooling2D(strides=(nb_pools, nb_pools), padding='same', data_format="channels_last"))
    
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Conv2D(conv_layer_sizes[1], (nb_kernels, nb_kernels), activation='relu'))
    model.add(layers.MaxPooling2D(strides=(nb_pools, nb_pools),padding='same', data_format="channels_last"))
    
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Conv2D(conv_layer_sizes[2], (nb_kernels, nb_kernels), activation='relu'))
    model.add(layers.MaxPooling2D(strides=(nb_pools, nb_pools), padding='same', data_format="channels_last"))

    ## add the model on top of the convolutional base
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(128))
    #model.add(layers.Activation('relu'))
    model.add(layers.Dense(64))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(32))
    model.add(layers.Activation('relu'))
    #model.add(Dense(nb_classes))
    #model.add(Activation('softmax'))

    
    # the last layer is dependent on model type
    if model_type == 'regression':
        model.add(layers.Dense(units=1))
    else:
        model.add(layers.Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.0001),
                      metrics=['accuracy'])
    
    return model

# instantiate the model
conv_layer_sizes = (36, 128, 64)
dense_layer_size = 24
model = build_model('regression', conv_layer_sizes, dense_layer_size)
model.summary()

model_path =  './aaindex_checkpoints.h5'
checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
        save_best_only=True, mode='min')
# reduce_on_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=150)
callbacks_list = [checkpoint, early_stopping]

# build a model
model = build_model(model_type='regression', conv_layer_sizes=conv_layer_sizes,
        dense_layer_size=dense_layer_size, dropout_rate=0.5)
model.compile(loss=metrics.mean_squared_error, optimizer=optimizers.Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    amsgrad=False
    ),
    metrics=[metrics.RootMeanSquaredError(name = 'rmse')]
)

model.fit(X_train, np.array(y_train), validation_data=(X_test, np.array(y_test)), 
                epochs=5000, batch_size=500, callbacks=callbacks_list, verbose=1)
model.save("aaindex_model.h5")
with open('./trainHistory_rmse.pickle', 'wb') as fp:
    pickle.dump(model.history.history, fp)
