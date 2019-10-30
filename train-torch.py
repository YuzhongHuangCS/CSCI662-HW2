import argparse
from model import PyTorchModel
import pdb
import pickle
from collections import Counter, defaultdict
import os
import numpy as np
import math

if __name__ == "__main__":
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-u', help='number of hidden units', type=int, default=100)
	parser.add_argument('-l', help='learning rate', type=float, default=0.01)
	parser.add_argument('-f', help='number of words to read per data item', type=int, default=1000)
	parser.add_argument('-b', help='minibatch size', type=int, default=1000000)
	parser.add_argument('-e', help='number of epochs to train for', type=int, default=1000)
	parser.add_argument('-E', help='word embedding file to be read', type=str, default='glove.6B.50d.txt')
	parser.add_argument('-i', help='training file to be read', type=str, default='4dim.train.txt')
	parser.add_argument('-o', help='model file to be written', type=str, default='model.pickle')
	parser.add_argument('-s', help='score mode', type=bool, default=False)
	parser.add_argument('-l2', help='l2 reg', type=float, default=1e-5)
	args = parser.parse_args()
	print('Args:', args)

	model = PyTorchModel(args)
	if args.s:
		model.score(args.i)
	else:
		model.train(args.i)
		with open(args.o, 'wb') as fout:
			pickle.dump(model, fout, pickle.HIGHEST_PROTOCOL)

	print('OK')
	print('123')
