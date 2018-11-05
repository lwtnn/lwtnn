#!/usr/bin/env python3

from keras.layers import Layer
from keras import backend as K
from keras.layers import initializers, InputSpec

class Swish(Layer):
    """
    Swish activation function with a trainable parameter referred to as 'beta' in https://arxiv.org/abs/1710.05941
    but as 'alpha' here."""
    def __init__(self, trainable_beta = True, beta_initializer = 'ones', **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable_beta
        self.beta_initializer = initializers.get(beta_initializer)
        self.__name__ = 'swish'
        
    def build(self, input_shape):
        self.beta = self.add_weight(shape=[1], name='beta', 
                                    initializer=self.beta_initializer)
        self.input_spec = InputSpec(ndim=len(input_shape))
        self.built = True

    def call(self, inputs):
        return inputs * K.sigmoid(self.beta * inputs)

    def get_config(self):
        config = {'trainable_beta': self.trainable, 
                  'beta_initializer': initializers.serialize(self.beta_initializer)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))