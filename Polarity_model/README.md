## Review Polarity Classifier using TransformerEncoder

The repository contains data and preprocessing modules required by the ipython notebook. The basic idea is to display 
the use of the TransformerEncoder as a classification model. The model is inspired by [BERT model](https://arxiv.org/pdf/1810.04805.pdf).
Pretrained Glove vectors have been taken as input and first index has been sliced to give representation of the full sentence.
A sigmoid is then used for classification 

The model returns the attention weights of last TransformerEncoder which is later on visualised (Note : open in jupyter notebook to view)
