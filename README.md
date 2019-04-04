# Simple-TransformerEncoder-keras
A simple implementation of Transformer Encoder in keras based on [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). The implementation does not include masking. Trainable Positional embeddings is also added so that model could learn sequential information.
This repository also includes an example of Transformer as a classifier and its attention visualization.

## Usage
Transformer Encoder can be used as a layer in keras model with parameter:model dimension, heads, query_vector dimension, value_vector dimension, positional_ff_dim and dropouts

```
transformer_out, attention_weights=TransformerEncoder( d_model,
                                                       heads,
                                                       query_dim,
                                                       value_dim,
                                                       positional_ff_dim,
                                                       dropout_rate= 0.1,
                                                       name= 'TransformerEncoder')(input)
```

Attention weights of all heads are return for visualisation. TransformerEncoder can be stacked to make deep network as in the following example
```
attention=[None for i in range(transformer_depth)]
for i in range(transformer_depth):
        next_step_input,attention[i] =TransformerEncoder(word_embeddings_dim,
                                                         heads,
                                                         query_dim,
                                                         value_dim,
                                                         positional_ff_dim,
                                                         dropout_rate= 0.1,
                                                         name= 'Transformer'+str(i))(next_step_input)

```


## TODO :
1. Implement masking for all the subsequent layers in the Transformer block. (Keras requires masking for the layers otherwise raises an exception). A workaround has been implemented so that model does not attend on the padding spaces in Mask sub folder.
2. A better visualisation of the attention weights

Important: This is not the exact implementation of vanilla Transformer described in the paper, basic ideas of the Transformer block (multiheaded self attention using query, key and value , positional feed forward layer and Layer normalization) is coded with few changes. The layers have been implemented with a combination of lot of `Lambda`, so `model.summary()` will return a large block 😁
