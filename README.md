# tf_keras_quantization_experiment

## Background
While adding quatize awareness code in other porject by using tensorflow_model_optimization, I faced a lot of problem. To simplify all the problems I met, I made a simple sample code to simulate the problem I face on. I would like to share how I solved the problem base on this simply code to every one.

## Introduction
While creating or edit deep learning network in large project, whole network structure not always put in the same python code. Author sometime chop the network in pieces in different .py file and then using a merge function or code combines together. Another interesting things is that some library, like Keras, can build a model with multiple ways. 
For instance, 
1. we can temperary build the part of the models in tf.keras.Model and use it as an function to continue the rest of the model.
2. we can also put parts of the nodes in to tf.keras.Sequential, and then connect the "sequence" to other node.

Although both of the method are very convenient in model connection, it cannot work functionally in tensorflow_model_optimization if we try to add quantization awareness node inside the model. Therefore, I set a easy sample code to dig the both way to solve the problem. 

## Envirnment 
python3.8
tensorflow==2.4.0
tensorflow_model_optimization=0.5.0
