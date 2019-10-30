import sys
import pdb
from collections import Counter, defaultdict
import numpy as np
import math
from functools import partial
import tempfile
import os
import pickle
import torch
import torch.nn.functional as F
import ops

np.random.seed(1234)
torch.manual_seed(1234)

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
		#with open(pickle_filename, 'wb') as fout:
		#	pickle.dump(word_embedding, fout, pickle.HIGHEST_PROTOCOL)

		return word_embedding

class Model(object):
	def __init__(self, args):
		super(Model, self).__init__()
		self.args = args

	def score(self, inputfile):
		all_lines = open(inputfile, encoding='utf-8').readlines()
		np.random.shuffle(all_lines)

		n_total = len(all_lines)
		n_test = int(n_total * 0.1)
		n_train = n_total - n_test

		fid, train_inputfile = tempfile.mkstemp()
		fid, test_inputfile = tempfile.mkstemp()
		fid, test_outpufile = tempfile.mkstemp()

		with open(train_inputfile, 'w', encoding='utf-8') as fout:
			for line in all_lines[:n_train]:
				fout.write(line)

		with open(test_inputfile, 'w', encoding='utf-8') as fout:
			for line in all_lines[n_train:]:
				fout.write(line)

		self.train(train_inputfile)
		self.test(test_inputfile, test_outpufile)

		true_labels = np.asarray([line.rstrip().split('\t')[1] for line in open(test_inputfile, encoding='utf-8')])
		pred_labels = np.asarray([line.rstrip() for line in open(test_outpufile, encoding='utf-8')])
		acc = np.mean(true_labels == pred_labels)
		print('Test Accuracy', acc)

	def prepare_train_data(self, inputfile):
		args = self.args
		label_counter = Counter()
		embedding = initialize_embedding(args.E)
		if 'urdu' in args.E:
			unk = initialize_embedding('unk-urdu.vec')['UNK']
		else:
			unk = initialize_embedding('unk.vec')['UNK']
		embedding_dim = len(unk)
		X_all = []
		Y_all_text = []
		for line in open(inputfile, encoding='utf-8'):
			text, truth = line.rstrip().split('\t')
			label_counter.update([truth])
			if len(text) == 0:
				emb = np.zeros((args.f, embedding_dim))
			else:
				emb = [embedding.get(t.lower(), unk) for t in text.split()[:args.f]]
			emb = np.pad(emb, ((0, args.f - len(emb)), (0, 0)), mode='constant', constant_values=0)
			X_all.append(emb.flatten())
			Y_all_text.append(truth)

		label_map = {value: index for index, value in enumerate(label_counter.keys())}
		label_inv_map = {index: value for index, value in enumerate(label_counter.keys())}
		Y_all = [label_map[y] for y in Y_all_text]
		Y_inv_all = [label_inv_map[v] for v in Y_all]

		indices_all = list(range(len(Y_all)))
		np.random.shuffle(indices_all)

		n_total = len(indices_all)
		n_train = int(n_total * 0.9)
		n_valid = n_total - n_train

		indices_train = indices_all[:n_train]
		indices_valid = indices_all[n_train:]

		X_all = np.asarray(X_all, dtype=np.float32)
		Y_all = np.asarray(Y_all, dtype=np.int64)
		X_train = X_all[indices_train]
		Y_train = Y_all[indices_train]
		X_valid = X_all[indices_valid]
		Y_valid = Y_all[indices_valid]

		self.label_map = label_map
		self.label_inv_map = label_inv_map

		return X_train, Y_train, X_valid, Y_valid, embedding_dim

	def prepare_test_data(self, inputfile):
		args = self.args
		embedding = initialize_embedding(args.E)
		if 'urdu' in args.E:
			unk = initialize_embedding('unk-urdu.vec')['UNK']
		else:
			unk = initialize_embedding('unk.vec')['UNK']

		X_all = []
		for line in open(inputfile, 'r', encoding='utf-8'):
			text = line.rstrip().split('\t')
			#support line with label and without label
			if isinstance(text, list):
				text = text[0]

			if len(text) == 0:
				emb = np.zeros((args.f, len(unk)))
			else:
				emb = [embedding.get(t.lower(), unk) for t in text.split()[:args.f]]
			emb = np.pad(emb, ((0, args.f - len(emb)), (0, 0)), mode='constant', constant_values=0)
			X_all.append(emb.flatten())
		return X_all

	def train(self, inputfile):
		pass

	def test(self, inputfile, outputfile):
		pass


class PyTorchModel(Model):
	"""docstring for PyTorchModel"""
	def __init__(self, args):
		super(PyTorchModel, self).__init__(args)
		self.is_training = False

	def net(self, X):
		if self.is_training:
			X = F.dropout(X, p=0.25)
		h = F.sigmoid(torch.matmul(X, self.WA) + self.bA)
		if self.is_training:
			h = F.dropout(h, p=0.4)
		l = torch.matmul(h, self.WB) + self.bB
		return l

	def save_weight(self):
		self.WA_clone = self.WA.clone()
		self.bA_clone = self.bA.clone()
		self.WB_clone = self.WB.clone()
		self.bB_clone = self.bB.clone()


	def load_weight(self):
		self.WA = self.WA_clone.clone()
		self.bA = self.bA_clone.clone()
		self.WB = self.WB_clone.clone()
		self.bB = self.bB_clone.clone()

	def train(self, inputfile):
		args = self.args
		X_train, Y_train, X_valid, Y_valid, embedding_dim = self.prepare_train_data(inputfile)
		X_valid = torch.from_numpy(X_valid)
		Y_valid = torch.from_numpy(Y_valid)

		WA = torch.from_numpy(np.random.normal(0, 1, (embedding_dim*args.f, args.u)).astype(np.float32))
		WA.requires_grad = True
		bA = torch.from_numpy(np.random.normal(0, 1, (1, args.u)).astype(np.float32))
		bA.requires_grad = True

		WB = torch.from_numpy(np.random.normal(0, 1, (args.u, len(self.label_map))).astype(np.float32))
		WB.requires_grad = True
		bB = torch.from_numpy(np.random.normal(0, 1, (1, len(self.label_map))).astype(np.float32))
		bB.requires_grad = True

		self.WA = WA
		self.bA = bA
		self.WB = WB
		self.bB = bB

		n_train = len(X_train)
		indices_train = list(range(n_train))
		batches = math.ceil(n_train / args.b)
		print('Batches', batches)
		opt = torch.optim.Adam([WA, bA, WB, bB], lr=args.l, weight_decay=args.l2)

		smallest_valid_loss = float('inf')
		best_acc = 0
		wait = 0
		n_lr_decay = 5
		n_break = 20

		#train_s = []
		#valid_s = []
		for e in range(args.e):
			np.random.shuffle(indices_train)
			nll_train_ary = []
			nll_valid_ary = []
			acc_train_ary = []
			acc_valid_ary = []
			for b in range(batches):
				this_indices = indices_train[b * args.b : (b+1) * args.b]
				X = torch.from_numpy(X_train[this_indices])
				Y = torch.from_numpy(Y_train[this_indices])

				self.is_training = True
				l = self.net(X)
				nll = F.cross_entropy(l, Y)
				pred = np.argmax(l.detach().numpy(), axis=-1)
				acc = np.mean(pred == Y.detach().numpy())

				self.is_training = False
				l_valid = self.net(X_valid)
				nll_valid = F.cross_entropy(l_valid, Y_valid)
				pred_valid = np.argmax(l_valid.detach().numpy(), axis=-1)
				acc_valid = np.mean(pred_valid == Y_valid.detach().numpy())

				opt.zero_grad()
				nll.backward()
				opt.step()

				nll_train_ary.append(nll.detach().numpy())
				nll_valid_ary.append(nll_valid.detach().numpy())
				acc_train_ary.append(acc)
				acc_valid_ary.append(acc_valid)

			#train_s.append(nll_np)
			#valid_s.append(nll_valid_np)

			nll_train_np = np.mean(nll_train_ary)
			nll_valid_np = np.mean(nll_valid_ary)
			acc_train = np.mean(acc_train_ary)
			acc_valid = np.mean(acc_valid_ary)

			print(f'Epoch: {e}, Train NLL: {nll_train_np}, Train Acc: {acc_train}, Valid NLL: {nll_valid_np}, Valid Acc: {acc_valid}')

			if best_acc < acc_valid:
				best_acc = acc_valid
				smallest_valid_loss = nll_valid_np
				self.save_weight()
				wait = 0
				print('New smallest')
			else:
				wait += 1
				print('Wait {}'.format(wait))
				if wait % n_lr_decay == 0:
					opt.param_groups[0]['lr'] *= 0.95
					print('Apply lr decay, new lr: %f' % opt.param_groups[0]['lr'])

				if wait % n_break == 0:
					print('Break')
					break

		self.load_weight()
		print('Best valid acc', best_acc)
		'''
		import matplotlib.pyplot as plt
		plt.plot(train_s, label='train')
		plt.plot(valid_s, label='valid')
		plt.xlabel('Epoch')
		plt.ylabel('NLL')
		plt.ylim(0, 50)
		plt.title('PyTorch')
		plt.legend()
		#pdb.set_trace()
		'''


	def test(self, inputfile, outputfile):
		X_all = self.prepare_test_data(inputfile)
		X = torch.from_numpy(np.asarray(X_all, dtype=np.float32))

		self.is_training = False
		l = self.net(X)
		pred = np.argmax(l.detach().numpy(), axis=-1)
		pred_text = [self.label_inv_map[index] for index in pred]
		with open(outputfile, 'w') as fout:
			fout.write('\n'.join(pred_text))

class NumpyModel(Model):
	"""docstring for NumpyModel"""
	def __init__(self, args):
		super(NumpyModel, self).__init__(args)
		self.is_training = False

	def net(self, X):
		if self.is_training:
			X = ops.dropout(X, drop_prob=0.1)
		h = ops.relu(ops.add(ops.matmul(X, self.WA), self.bA))
		if self.is_training:
			h = ops.dropout(h, drop_prob=0.1)
		l = ops.add(ops.matmul(h, self.WB), self.bB)
		return l

	def save_weight(self):
		self.WA_clone = self.WA.copy()
		self.bA_clone = self.bA.copy()
		self.WB_clone = self.WB.copy()
		self.bB_clone = self.bB.copy()


	def load_weight(self):
		self.WA = self.WA_clone.copy()
		self.bA = self.bA_clone.copy()
		self.WB = self.WB_clone.copy()
		self.bB = self.bB_clone.copy()

	def train(self, inputfile):
		args = self.args
		X_train, Y_train, X_valid, Y_valid, embedding_dim = self.prepare_train_data(inputfile)

		WA = ops.Tensor(np.random.normal(0, 1, (embedding_dim*args.f, args.u)).astype(np.float32), requires_grad=True)
		bA = ops.Tensor(np.random.normal(0, 1, (1, args.u)).astype(np.float32), requires_grad=True)

		WB = ops.Tensor(np.random.normal(0, 1, (args.u, len(self.label_map))).astype(np.float32), requires_grad=True)
		bB = ops.Tensor(np.random.normal(0, 1, (1, len(self.label_map))).astype(np.float32), requires_grad=True)

		self.WA = WA
		self.bA = bA
		self.WB = WB
		self.bB = bB

		n_train = len(X_train)
		indices_train = list(range(n_train))
		batches = math.ceil(n_train / args.b)
		print('Batches', batches)
		opt = ops.AdamOptimizer([WA, bA, WB, bB], args.l)

		smallest_valid_loss = float('inf')
		best_acc = 0
		wait = 0
		n_lr_decay = 5
		n_break = 20

		#train_s = []
		#valid_s = []
		for e in range(args.e):
			np.random.shuffle(indices_train)
			nll_train_ary = []
			nll_valid_ary = []
			acc_train_ary = []
			acc_valid_ary = []
			for b in range(batches):
				this_indices = indices_train[b * args.b : (b+1) * args.b]
				X = ops.Tensor(X_train[this_indices])
				Y = ops.Tensor(Y_train[this_indices])

				self.is_training = True
				l = self.net(X)
				nll = ops.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=l)
				pred = np.argmax(l, axis=-1)
				acc = np.mean(pred == Y)

				self.is_training = False
				l_valid = self.net(X_valid)
				nll_valid = ops.sparse_softmax_cross_entropy_with_logits(labels=Y_valid, logits=l_valid)
				pred_valid = np.argmax(l_valid, axis=-1)
				acc_valid = np.mean(pred_valid == Y_valid)

				l2_WA = ops.l2_loss(WA, args.l2)
				l2_bA = ops.l2_loss(bA, args.l2)
				l2_WB = ops.l2_loss(WB, args.l2)
				l2_bB = ops.l2_loss(bB, args.l2)

				opt.zero_grad()
				nll.backward()
				l2_WA.backward()
				l2_bA.backward()
				l2_WB.backward()
				l2_bB.backward()
				opt.step()

				nll_train_ary.append(nll)
				nll_valid_ary.append(nll_valid)
				acc_train_ary.append(acc)
				acc_valid_ary.append(acc_valid)


			nll_train_np = np.mean(nll_train_ary)
			nll_valid_np = np.mean(nll_valid_ary)
			acc_train = np.mean(acc_train_ary)
			acc_valid = np.mean(acc_valid_ary)

			print(f'Epoch: {e}, Train NLL: {nll_train_np}, Train Acc: {acc_train}, Valid NLL: {nll_valid_np}, Valid Acc: {acc_valid}')

			if best_acc < acc_valid:
				best_acc = acc_valid
				smallest_valid_loss = nll_valid_np
				self.save_weight()
				wait = 0
				print('New smallest')
			else:
				wait += 1
				print('Wait {}'.format(wait))
				if wait % n_lr_decay == 0:
					opt.lr *= 0.95
					print('Apply lr decay, new lr: %f' % opt.lr)

				if wait % n_break == 0:
					print('Break')
					break

		self.load_weight()
		print('Best valid acc', best_acc)

		'''
		import matplotlib.pyplot as plt
		plt.plot(train_s, label='train')
		plt.plot(valid_s, label='valid')
		plt.xlabel('Epoch')
		plt.ylabel('NLL')
		plt.ylim(0, 50)
		plt.title('Numpy')
		plt.legend()
		pdb.set_trace()
		'''


	def test(self, inputfile, outputfile):
		X_all = self.prepare_test_data(inputfile)
		X = ops.Tensor(np.asarray(X_all, dtype=np.float32))
		h_raw = ops.add(ops.matmul(X, self.WA), self.bA)
		h = ops.relu(h_raw)
		#h_c = ops.relu(ops.add(ops.matmul(h, self.WC), self.bC))

		l = ops.add(ops.matmul(h, self.WB), self.bB)
		pred = np.argmax(l, axis=-1)
		pred_text = [self.label_inv_map[index] for index in pred]
		with open(outputfile, 'w') as fout:
			fout.write('\n'.join(pred_text))
