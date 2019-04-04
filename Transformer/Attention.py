from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Permute,TimeDistributed,Dense,Reshape,Activation,Lambda,Dropout
import math
import numpy as np


class MultiHeadedAttention():
    '''
    Self-Attention layer of Transformer block described in "Attention is all you need".
    d_model: the output dimensions for embeddings of each token in the sequence.
    heads  : number of heads 
    dim_q  : query vector and key vector dimensions
    dim_v  : value vector dimensions
    dropout_rate : dropout rate before fully connected layer
    '''
    def __init__(self,d_model ,heads,dim_q,dim_v,dropout_rate,name, **kwargs):
        self.heads=heads
        self.dim_q=dim_q
        self.dim_v=dim_q

        self.query_layer   = TimeDistributed(Dense(self.heads*self.dim_q,use_bias = False),name = name+'_QueryLayer')
        self.key_layer     = TimeDistributed(Dense(self.heads*self.dim_q,use_bias = False),name = name+'_KeyLayer')
        self.value_layer   = TimeDistributed(Dense(self.heads*self.dim_v,use_bias = False),name = name+'_ValueLayer') 

        self.dropout_layer = Dropout(dropout_rate)

        self.output_layer  = TimeDistributed(Dense(d_model,use_bias = False),name = name+'_AttentionOutputLayer')

# moves up the timestep dimention towards the end and heads towards batch_size dimention
    def shift_timestep_ahead(self,vec,last_dim,time_steps,transpose=False):
      vec     = Reshape([-1,self.heads,last_dim])(vec)

      if transpose:
        vec   = Permute([2,3,1])(vec)
        vec   = Lambda(lambda x:K.reshape(x, [-1,  last_dim, time_steps ]))(vec)

      else:
        vec   = Permute([2,1,3])(vec)
        vec   = Lambda(lambda x:K.reshape(x, [-1, time_steps,  last_dim ]))(vec)

      return vec

# moves down the timestep dimention towards the batch_size dimention
    def shift_timestep_behind(self,vec,last_dim,time_steps):
      vec    = Lambda(lambda x:K.reshape(x, [-1,self.heads, time_steps,  last_dim ]))(vec)
      vec    = Permute([2,1,3])(vec)
        
      vec    = Lambda(lambda x:K.reshape(x, [-1,time_steps,self.heads*last_dim ]))(vec)

      return vec

    def scaledDotAttention(self):
        '''
        Scaled dot attention : computes the attention score by dot product of query vector and 
        key vector. Sums the value vectors of tokens based on attention weights( after softmax of attention score )
        '''
      attention_score   = Lambda( lambda x: K.batch_dot(x[0],x[1]) / np.sqrt(self.dim_q))([self.query_vec,self.key_vec])
      attention_weights = Activation('softmax')(attention_score)
      attention_vector  = Lambda( lambda x: K.batch_dot(x[0],x[1]))([attention_weights,self.value_vec])

      return attention_vector, attention_weights

    def __call__(self, x):

        a = x
        time_steps = int(a.get_shape()[1])
        self.query_vec = self.query_layer(a)
        self.key_vec   = self.key_layer(a)
        self.value_vec = self.value_layer(a)

        # Reshape
        self.query_vec = self.shift_timestep_ahead(self.query_vec,self.dim_q,time_steps,transpose = False)
        self.key_vec   = self.shift_timestep_ahead(self.key_vec,self.dim_q,time_steps,transpose = True)
        self.value_vec = self.shift_timestep_ahead(self.value_vec,self.dim_v,time_steps,transpose= False)

        # Attention Weights
        self.attention_vec,self.attention_weights = self.scaledDotAttention()

        # print(self.attention_weights,self.attention_vec)
        self.attention_weights = Lambda( lambda x: K.reshape(x,[-1,self.heads,time_steps,time_steps]))(self.attention_weights)
        self.attention_vec     = Lambda( lambda x: K.reshape(x,[-1,time_steps,self.dim_v]))(self.attention_vec)

        #Output Fully Connected Dense
        temp_out               = self.shift_timestep_behind(self.attention_vec,self.dim_v,time_steps)
        temp_out               = self.dropout_layer(temp_out)
        self.out_vec           = self.output_layer(temp_out)

        return [self.out_vec,self.attention_weights]

