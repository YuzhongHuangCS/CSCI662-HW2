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

	X = torch.from_numpy(np.asarray(X, dtype=np.float32))
	Y = torch.from_numpy(np.asarray(Y, dtype=np.int64))
	WA = torch.randn(embedding_dim*args.f, args.u, requires_grad=True)
	bA = torch.randn(1, args.u, requires_grad=True)
	relu = torch.nn.ReLU()
	entropy = torch.nn.CrossEntropyLoss()

	WB = torch.randn(args.u, len(label_map), requires_grad=True)
	bB = torch.randn(1, len(label_map), requires_grad=True)


	optimizer = torch.optim.Adam([WA, bA, WB, bB], lr=args.l, weight_decay=1e-5)
	for z in range(100):
		h = relu(torch.matmul(X, WA) + bA)
		l = torch.matmul(h, WB) + bB
		loss = entropy(l, Y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(z, loss)

	pdb.set_trace()
	print('123')
