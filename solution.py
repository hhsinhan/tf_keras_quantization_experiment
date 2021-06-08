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

# Situation 2, set encoder as a tf.keras.Model object and use it as a function
# to continues connecting to decoder
encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")
x = tf.keras.layers.Reshape((4, 4, 1))(encoder.output)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = tf.keras.layers.UpSampling2D(3)(x)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output2 = tf.keras.layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder2 = tf.keras.Model(encoder_input, decoder_output2, name="autoencoder2")
autoencoder2.summary()

# Situation 3, set a tf.keras.Sequential and re-write decoder inside the object.
# The combine the encoder to the sequential decoder.
decoder_sequential = tf.keras.Sequential()
decoder_sequential.add(tf.keras.layers.Reshape((4, 4, 1)))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(16, 3, activation="relu"))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(32, 3, activation="relu"))
decoder_sequential.add(tf.keras.layers.UpSampling2D(3))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(16, 3, activation="relu"))
decoder_sequential.add(tf.keras.layers.Conv2DTranspose(1, 3, activation="relu"))
for i in range(len(decoder_sequential.layers)):
  if i == 0:
    decoder_output3 = decoder_sequential.layers[i](encoder_output)
  else:
    decoder_output3 = decoder_sequential.layers[i](decoder_output3)


#autoencoder3 = tf.keras.Model(encoder_input, decoder_sequential(encoder_output), name="autoencoder3")
autoencoder3 = tf.keras.Model(encoder_input, decoder_output3, name="autoencoder3")
autoencoder3.summary()

"""
# TODO add quantization awareness node function in all situations and print out node for them
# q_aware_model = tfmo.quantization.keras.quantize_model(model_mv2) # this runs













