from keras.engine import Layer
import keras.backend as K
import tensorflow as tf
class MaskLayer(Layer):
  """
  Generates the mask for the input layer
  """
  
  def __init__(self,mask_function=None, **kwargs):
      super().__init__(**kwargs)

    
  def build(self,input_shape):
    super().build(input_shape)
    
    
  def call(self, inputs, **kwargs):

      isGreater = K.cast(K.greater(inputs,0),'float32')
      result = (isGreater)
      return result
    