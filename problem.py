# import tensorflow & tensorflow_model_optimization
import tensorflow_model_optimization as tfmo
import tensorflow as tf

"""
This sample code took autoencoder as an example. 
Basically, autoencoder is constructed by encode and decoder.
The experiments below try different ways to define these two part.   
"""

# Situation 1, normal connection

# encoder
encoder_input = tf.keras.Input(shape=(28, 28, 1), name="img")
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(3)(x)
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = tf.keras.layers.GlobalMaxPooling2D()(x)

x = tf.keras.layers.Reshape((4, 4, 1))(encoder_output)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = tf.keras.layers.UpSampling2D(3)(x)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = tf.keras.layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = tf.keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
"""
Model: "autoencoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
img (InputLayer)             [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 16)        160       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        4640      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 32)          9248      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 16)          4624      
_________________________________________________________________
global_max_pooling2d (Global (None, 16)                0         
_________________________________________________________________
reshape (Reshape)            (None, 4, 4, 1)           0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 6, 6, 16)          160       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 8, 8, 32)          4640      
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 24, 24, 32)        0         
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 26, 26, 16)        4624      
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 28, 28, 1)         145       
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________
"""

# Situation 2, set encoder as a tf.keras.Model object and use it as a function
# to continues connecting to decoder
encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")
encoder_output2 = encoder(encoder_input)
x = tf.keras.layers.Reshape((4, 4, 1))(encoder_output2)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = tf.keras.layers.UpSampling2D(3)(x)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output2 = tf.keras.layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder2 = tf.keras.Model(encoder_input, decoder_output2, name="autoencoder2")
autoencoder2.summary()

"""
Model: "autoencoder2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
img (InputLayer)             [(None, 28, 28, 1)]       0         
_________________________________________________________________
encoder (Functional)         (None, 16)                18672     
_________________________________________________________________
reshape_1 (Reshape)          (None, 4, 4, 1)           0         
_________________________________________________________________
conv2d_transpose_4 (Conv2DTr (None, 6, 6, 16)          160       
_________________________________________________________________
conv2d_transpose_5 (Conv2DTr (None, 8, 8, 32)          4640      
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 24, 24, 32)        0         
_________________________________________________________________
conv2d_transpose_6 (Conv2DTr (None, 26, 26, 16)        4624      
_________________________________________________________________
conv2d_transpose_7 (Conv2DTr (None, 28, 28, 1)         145       
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________
"""

# Situation 3, set a tf.keras.Sequential and re-write decoder inside the object.
# The combine the encoder to the sequential decoder.
decoder_sequential = tf.keras.Sequential()
decoder_sequential.add(tf.keras.layers.Reshape((4, 4, 1)))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(16, 3, activation="relu"))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(32, 3, activation="relu"))
decoder_sequential.add(tf.keras.layers.UpSampling2D(3))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(16, 3, activation="relu"))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(1, 3, activation="relu"))

autoencoder3 = tf.keras.Model(encoder_input, decoder_sequential(encoder_output), name="autoencoder3")
autoencoder3.summary()

"""
Model: "autoencoder3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
img (InputLayer)             [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 16)        160       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        4640      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 6, 32)          9248      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 16)          4624      
_________________________________________________________________
global_max_pooling2d (Global (None, 16)                0         
_________________________________________________________________
sequential (Sequential)      (None, 26, 26, 16)        9424      
=================================================================
Total params: 28,096
Trainable params: 28,096
Non-trainable params: 0
_________________________________________________________________
"""
# TODO add quantization awareness node function in all situations and print out node for them
# q_aware_model = tfmo.quantization.keras.quantize_model(model_mv2) # this runs













