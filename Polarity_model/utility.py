import pickle
import numpy as np

def read_data(filename):
	"""
	Reads data from file with TREC format.
	Return seperated X data and y data

	"""
	X=[]
	y=[]
	with open(filename,'r') as file:
	    for line in file:
	        tokens = line.strip().split(':')
	        question = " ".join(tokens[1:])
	        X.append(" ".join(question.split(" ")[1:]))
	        y.append(tokens[0])
	    
	return X,y


def load_data_from_pickle(filepath):
	"""
	Loads the data from the pickle file created earlier

	"""
	all_data = pickle.load(open(filepath, 'rb'))

	data = all_data['data']
	labels = all_data['labels']
	label_names = all_data['label_names']
	val = all_data['imdb']
	val_labels = all_data['imdb_labels']
	
	return data,labels,label_names,val,val_labels

def loadGlove(filename):
	"""
	Reads the pre-trained word embeddings from 
	a text file, with each dimenstion separated by space
	"""
	glove=dict()
	with open(filename,'r',encoding = 'utf-8') as glovefile:
	    for line in glovefile:
	        temp = line.split()
	        word = temp[0]
	        glove[word] = list(map(float,temp[1:]))
	return glove


def getEmbMatrix(glove,word2index,dim):
	"""
	Generates a embeddings matrix for the Model
	Args:
    glove: Dictionary with each word as key and word- vector as value.
    word2index : A mapping for each word to a index. Generated during pre-
    			 processing
    dim        : dimension of the word-embeddings
  	Returns:
    a 2d matrix filled with word-vector corresponding to word-index.
	"""

	emb_mat = np.zeros((len(word2index)+1, dim))
	print(emb_mat.shape)
	for word in word2index.keys():
	    if word in glove:
	        emb_mat[word2index[word]] = glove[word]

	return emb_mat


