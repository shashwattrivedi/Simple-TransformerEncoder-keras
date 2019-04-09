from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Add
import math
from Attention import MultiHeadedAttention
from LayerNormalization import LayerNormalization
from PositionWiseFeedForward import PositionWiseFeedForward

class TransformerEncoder():
    '''
    Main Transformer Encoder block : Encapsulates different layers with a Transformer Encoder block and calls them in order.
    Inputs
    d_model : dimensions of the output and internal layers
    heads   : number of heads
    dim_q   : query and key dimension 
    dim_v   : value dimension
    hidden_units : hidden units for the positionwise feed forward network
    dropout_rate : dropout_rate

    Outputs
    A tuple:Transformer Encoder Representation, attention weights for each head and token
    '''
    def __init__(self, d_model,heads,dim_q,dim_v,hidden_units,dropout_rate,name,activation='relu', **kwargs):
        self.dim_v        = dim_v
        self.dim_q        = dim_q
        self.hidden_units = hidden_units
        self.heads        = heads

        self.attention_layer      = MultiHeadedAttention(d_model = d_model,heads = self.heads,dim_q = self.dim_q,dim_v = self.dim_v,dropout_rate=dropout_rate,name=name)
        self.normalization_layer  = LayerNormalization()
        self.feedforward          = PositionWiseFeedForward(d_model = d_model,inner_dim = self.hidden_units,dropout_rate=dropout_rate,name=name)


    def __call__(self, x):

        attention_vec,attention_weights   = self.attention_layer(x)
        normalized_inp                    = self.normalization_layer(Add()([attention_vec,x]))
        feedforward_out                   = self.feedforward(normalized_inp)

        transformer_out = self.normalization_layer(Add()([feedforward_out,normalized_inp]))

        return [transformer_out,attention_weights]

