import argparse
from model import Model
import pdb
import pickle
from collections import Counter, defaultdict
import os
import numpy as np
import torch
from functools import partial

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

class Tensor(object):
	def __init__(self, value, requires_grad=False):
		super(Tensor, self).__init__()
		self.value = value
		self.grad = None
		self.requires_grad = requires_grad

	def backward(self, value):
		self.grad = value

def matmul(t1, t2):
	newvalue = np.matmul(t1.value, t2.value)
	newtensor = Tensor(newvalue, requires_grad=True)
	def backward(value):
		g1 = None
		g2 = None
		if t1.requires_grad:
			g1 = value.dot(t2.value.T)
			t1.backward(g1)
		if t2.requires_grad:
			g2 = t1.value.T.dot(value)
			t2.backward(g2)

		newtensor.grad = [g1, g2]

	newtensor.backward = backward
	return newtensor

def add(t1, t2):
	newvalue = t1.value + t2.value
	newtensor = Tensor(newvalue, requires_grad=True)
	def backward(value):
		g1 = None
		g2 = None
		if t1.requires_grad:
			g1 = value
			t1.backward(g1)
		if t2.requires_grad:
			g2 = np.mean(value, axis=0)
			t2.backward(g2)
		newtensor.grad = [g1, g2]
	newtensor.backward = backward
	return newtensor

def relu(t1):
	mask = (t1.value > 0).astype(np.int32)
	newvalue = np.maximum(t1.value, 0.0)
	newtensor = Tensor(newvalue, requires_grad=True)

	def backward(value):
		g1 = None
		if t1.requires_grad:
			g1 = value * mask
			t1.backward(g1)
		newtensor.grad = g1
	newtensor.backward = backward
	return newtensor

def softmax_cross_entropy_with_logits(onehot, logits):
	l_softmax = softmax(logits.value)+1e-64
	nll = -np.mean(np.sum(np.log(l_softmax) * onehot, axis=-1))
	newtensor = Tensor(nll, requires_grad=True)
	def backward():
		delta = l_softmax - Y_onehot
		logits.backward(delta)
	newtensor.backward = backward
	return newtensor

def sgd(varlist, lr):
	for v in varlist:
		v.value -= v.grad * lr

if __name__ == "__main__":
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-u', help='number of hidden units', type=int, default=100)
	parser.add_argument('-l', help='learning rate', type=float, default=0.1)
	parser.add_argument('-f', help='number of words to read per data item', type=int, default=20)
	parser.add_argument('-b', help='minibatch size', type=int, default=100)
	parser.add_argument('-e', help='number of epochs to train for', type=int, default=100)
	parser.add_argument('-E', help='word embedding file to be read', type=str, default='glove.6B.50d.txt')
	parser.add_argument('-i', help='training file to be read', type=str, default='4dim.train.txt')
	parser.add_argument('-o', help='model file to be written', type=str, default='model.pickle')
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

	X = Tensor(np.asarray(X, dtype=np.float32))
	Y = Tensor(np.asarray(Y, dtype=np.int32))
	WA = Tensor(np.random.normal(0, 1, (embedding_dim*args.f, args.u)), requires_grad=True)
	bA = Tensor(np.random.normal(0, 1, (1, args.u)), requires_grad=True)

	WB = Tensor(np.random.normal(0, 1, (args.u, len(label_map))), requires_grad=True)
	bB = Tensor(np.random.normal(0, 1, (1, len(label_map))), requires_grad=True)


	for i in range(args.e):
		h_raw = add(matmul(X, WA), bA)
		h = relu(h_raw)
		l = add(matmul(h, WB), bB)

		nll = softmax_cross_entropy_with_logits(Y_onehot, l)
		print(i, nll.value)
		nll.backward()
		sgd([WA, bA, WB, bB], args.l)


	pdb.set_trace()
	print('123')
