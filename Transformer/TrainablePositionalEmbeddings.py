from keras.engine import Layer
class TransformerPositionalEmbedding(Layer):
    """
    Trainable positional embeddings: to be added to the inputs of Transformer block to learn 
    sequence information carried by the sentences.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        sequence_length, d_model = input_shape[-2:]
        self.position_embeddings = self.add_weight(
            shape=(sequence_length, d_model),
            initializer='uniform',
            name='word_position_embeddings',
            trainable=True)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        result = inputs + self.position_embeddings

        return result
