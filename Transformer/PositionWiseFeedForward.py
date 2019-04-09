from keras.layers import TimeDistributed,Dense,Dropout
import keras

class PositionWiseFeedForward():
    '''
        Applies a position wise feed forward dense layer on each token embedding
    '''
    def __init__(self,d_model,inner_dim,dropout_rate,name, **kwargs):

        self.inner_dim     = inner_dim

        self.hidden_layer  = TimeDistributed(Dense(self.inner_dim,activation=keras.activations.relu),name = name+'_PositionWiseFeedForward')
        self.dropout_layer = Dropout(dropout_rate)
        self.out_layer     = TimeDistributed(Dense(d_model),name = name+'_outffn')


    def __call__(self, x):

        self.hidden_vec = self.hidden_layer(x)
        self.hidden_vec = self.dropout_layer(self.hidden_vec)
        self.out_vec    = self.out_layer(self.hidden_vec)

        return self.out_vec

