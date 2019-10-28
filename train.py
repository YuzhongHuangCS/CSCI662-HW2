import argparse
from model import Model
import pdb
import pickle
from collections import Counter, defaultdict
import os
import numpy as np
import torch

def initialize_embedding(filename):
	pickle_filename = filename + '.pickle'
	if os.path.exists(pickle_filename):
		print('Loading embedding from cache')
		with open(pickle_filename, 'rb') as fin:
			return pickle.load(fin)
	else:
		word_embedding = {}
		print('Loading embedding from text')
		for line in open(filename, encoding='utf-8'):
			parts = line.rstrip().split(' ')
			word = parts[0]
			embs = [float(x) for x in parts[1:] if x]
			word_embedding[word] = embs
		with open(pickle_filename, 'wb') as fout:
			pickle.dump(word_embedding, fout, pickle.HIGHEST_PROTOCOL)

		return word_embedding

def softmax(ary):
	ary_exp = np.exp(ary-np.max(ary, axis=-1).reshape(-1, 1))
	return ary_exp / np.sum(ary_exp, axis=-1).reshape(-1, 1)

def delta_cross_entropy(X,y):
	"""
	X is the output from fully connected layer (num_examples x num_classes)
	y is labels (num_examples x 1)
		Note that y is not one-hot encoded vector.
		It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
	"""
	m = y.shape[0]
	grad = softmax(X)
	grad[range(m),y] -= 1
	grad = grad/m
	return grad

if __name__ == "__main__":
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-u', help='number of hidden units', type=int, required=True)
	parser.add_argument('-l', help='learning rate', type=float, required=True)
	parser.add_argument('-f', help='number of words to read per data item', type=int, required=True)
	parser.add_argument('-b', help='minibatch size', type=int, required=True)
	parser.add_argument('-e', help='number of epochs to train for', type=int, required=True)
	parser.add_argument('-E', help='word embedding file to be read', type=str, required=True)
	parser.add_argument('-i', help='training file to be read', type=str, required=True)
	parser.add_argument('-o', help='model file to be written', type=str, required=True)
	args = parser.parse_args()
	print('Args:', args)


	label_counter = Counter()
	embedding = initialize_embedding(args.E)
	unk = initialize_embedding('unk.vec')['UNK']
	embedding_dim = len(unk)
	X = []
	Y_text = []
	for line in open(args.i, encoding='utf-8'):
		text, truth = line.rstrip().split('\t')
		label_counter.update([truth])
		emb = [embedding.get(t, unk) for t in text.split()[:args.f]]
		emb = np.pad(emb, ((0, args.f - len(emb)), (0, 0)), mode='constant', constant_values=0)
		X.append(emb.flatten())
		Y_text.append(truth)

	label_map = {value: index for index, value in enumerate(label_counter.keys())}
	Y = [label_map[y] for y in Y_text]
	Y_onehot = np.zeros((len(Y), len(label_map)))
	for index, value in enumerate(Y):
		Y_onehot[index, value] = 1

	X = np.asarray(X, dtype=np.float32)
	Y = np.asarray(Y, dtype=np.int32)
	WA = np.random.normal(0, 1, (embedding_dim*args.f, args.u))
	bA = np.random.normal(0, 1, (1, args.u))

	WB = np.random.normal(0, 1, (args.u, len(label_map)))
	bB = np.random.normal(0, 1, (1, len(label_map)))


	for i in range(args.e):
		h_raw = np.matmul(X, WA) + bA
		mask = (h_raw > 0).astype(np.int32)
		h = np.maximum(h_raw, 0.0)
		l = np.matmul(h, WB) + bB
		l_softmax = softmax(l)+1e-64

		#pdb.set_trace()
		nll = -np.mean(np.sum(np.log(l_softmax) * Y_onehot, axis=-1))
		print(i, nll)

		lr = args.l
		delta = (l_softmax - Y_onehot) * lr
		delta_bB = np.mean(delta, axis=0)
		delta_WB = h.T.dot(delta)
		delta_h = delta.dot(WB.T)
		delta_h_raw = delta_h * mask
		delta_bA = np.mean(delta_h_raw, axis=0)
		delta_WA = X.T.dot(delta_h_raw)

		WA -= delta_WA
		bA -= delta_bA
		WB -= delta_WB
		bB -= delta_bB



	pdb.set_trace()
	print('123')
