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

class Tensor(np.ndarray):
	def __new__(cls, input_array, requires_grad=False):
		obj = np.asarray(input_array).view(cls)
		obj.grad = None
		obj.requires_grad = requires_grad
		return obj

	def backward(self, grad):
		self.grad = grad

def matmul(t1, t2):
	v = np.matmul(t1, t2)
	v.requires_grad = True

	def backward(grad):
		g1 = None
		g2 = None
		if t1.requires_grad:
			g1 = grad.dot(t2.T)
			t1.backward(g1)
		if t2.requires_grad:
			g2 = t1.T.dot(grad)
			t2.backward(g2)
		v.grad = [g1, g2]

	v.backward = backward
	return v

def add(t1, t2):
	v = t1 + t2
	v.requires_grad = True

	def backward(grad):
		g1 = None
		g2 = None
		if t1.requires_grad:
			g1 = grad
			t1.backward(g1)
		if t2.requires_grad:
			# hack for broadcasting
			if t2.squeeze().ndim == 1:
				g2 = np.mean(grad, axis=0)
			else:
				g2 = grad
			t2.backward(g2)
		v.grad = [g1, g2]
	v.backward = backward
	return v

def relu(t1):
	v = np.maximum(t1, 0.0)
	v.requires_grad = True
	mask = (t1 > 0).astype(np.int32)

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = grad * mask
			t1.backward(g1)
		v.grad = g1
	v.backward = backward
	return v

def sigmoid(t1):
	v = 1 / (1 + np.exp(-t1))
	v.requires_grad = True

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = v * (1 - v) * grad
			t1.backward(g1)
		v.grad = g1
	v.backward = backward
	return v

def tanh(t1):
	v = np.tanh(t1)
	v.requires_grad = True

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = (1 - v**2) * grad
			t1.backward(g1)
		v.grad = g1
	v.backward = backward
	return v

def dropout(t1, keep_prob):
	# scaled by 1/keep_prob
	mask = np.random.binomial(1, keep_prob, size=t1.shape) / keep_prob
	v = t1 * mask
	v.requires_grad = True

	def backward(grad):
		g1 = None
		if t1.requires_grad:
			g1 = mask * grad
			t1.backward(g1)
		v.grad = g1
	v.backward = backward
	return v

def sparse_softmax_cross_entropy_with_logits(labels, logits):
	l_softmax = softmax(logits) + np.finfo(logits.dtype).eps
	nll = -np.mean(np.log(l_softmax[range(len(labels)), labels]))
	nll.requires_grad = True

	def backward():
		l_softmax[range(len(labels)), labels] -= 1
		logits.backward(l_softmax)
	nll.backward = backward
	return nll

class SGDOptimizer(object):
	"""docstring for SGDOptimizer"""
	def __init__(self, var_list, lr, momentum=0.9, nesterov=False):
		super(SGDOptimizer, self).__init__()
		self.var_list = var_list
		self.lr = lr
		self.momentum = momentum
		self.v = None
		self.nesterov = nesterov

	def step(self):
		this_grad = [v.grad*self.lr for v in self.var_list]
		if self.v is None:
			new_v = this_grad
		else:
			new_v = [self.momentum * v + g for v, g in zip(self.v, this_grad)]
		self.v = new_v

		for var, v in zip(self.var_list, new_v):
			var -= v

		if self.nesterov:
			for var, v in zip(self.var_list, new_v):
				var -= self.momentum * v

class RMSPropOptimizer(object):
	"""docstring for RMSPropOptimizer"""
	def __init__(self, var_list, lr, decay_rate=0.9):
		super(RMSPropOptimizer, self).__init__()
		self.var_list = var_list
		self.lr = lr
		self.decay_rate = decay_rate
		self.r = None

	def step(self):
		this_r = [v.grad**2 * (1-self.decay_rate) for v in self.var_list]
		if self.r is None:
			new_r = this_r
		else:
			new_r = [self.decay_rate * old_r + new_r for old_r, new_r in zip(self.r, this_r)]
		self.r = new_r

		for var, r in zip(self.var_list, new_r):
			var -= (self.lr / np.sqrt(1e-6 + r)) * var.grad


class AdamOptimizer(object):
	def __init__(self, var_list, lr, p1=0.9, p2=0.999):
		super(AdamOptimizer, self).__init__()
		self.var_list = var_list
		self.lr = lr
		self.p1 = p1
		self.p2 = p2
		self.t = 1
		self.s = None
		self.r = None

	def step(self):
		this_s = [v.grad * (1-self.p1) for v in self.var_list]
		this_r = [v.grad**2 * (1-self.p2) for v in self.var_list]
		if self.s is None:
			new_s = this_s
		else:
			new_s = [self.p1 * old_s + new_s for old_s, new_s in zip(self.s, this_s)]
		if self.r is None:
			new_r = this_r
		else:
			new_r = [self.p2 * old_r + new_r for old_r, new_r in zip(self.r, this_r)]
		self.s = new_s
		self.r = new_r

		new_s = np.asarray(new_s) / (1-self.p1**self.t)
		new_r = np.asarray(new_r) / (1-self.p2**self.t)
		new_coef = [s/(np.sqrt(r) + 1e-8) for s, r in zip(new_s, new_r)]

		for var, coef in zip(self.var_list, new_coef):
			var -= self.lr * coef

		self.t += 1
if __name__ == "__main__":
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-u', help='number of hidden units', type=int, default=100)
	parser.add_argument('-l', help='learning rate', type=float, default=0.01)
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
	X = Tensor(np.asarray(X, dtype=np.float32))
	Y = Tensor(np.asarray(Y, dtype=np.int32))

	WA = Tensor(np.random.normal(0, 1, (embedding_dim*args.f, args.u)), requires_grad=True)
	bA = Tensor(np.random.normal(0, 1, (1, args.u)), requires_grad=True)

	Wh = Tensor(np.random.normal(0, 1, (args.u, args.u)), requires_grad=True)
	bh = Tensor(np.random.normal(0, 1, (1, args.u)), requires_grad=True)

	WB = Tensor(np.random.normal(0, 1, (args.u, len(label_map))), requires_grad=True)
	bB = Tensor(np.random.normal(0, 1, (1, len(label_map))), requires_grad=True)

	#opt = SGDOptimizer([WA, bA, Wh, bh, WB, bB], args.l, momentum=0)
	opt = RMSPropOptimizer([WA, bA, Wh, bh, WB, bB], args.l)
	for i in range(args.e):
		h_raw = add(matmul(X, WA), bA)
		h = tanh(h_raw)

		hh_raw = add(matmul(h, Wh), bh)
		hh = dropout(relu(hh_raw), 0.5)

		l = add(matmul(hh, WB), bB)

		nll = sparse_softmax_cross_entropy_with_logits(labels=Y, logits=l)
		pred = np.argmax(l, axis=-1)
		acc = np.mean(pred == Y)
		print(i, nll, acc)
		nll.backward()
		opt.step()
		#sgd([WA, bA, Wh, bh, WB, bB], args.l)


	pdb.set_trace()
	print('123')
