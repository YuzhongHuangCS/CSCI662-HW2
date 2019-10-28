import argparse
from model import Model
import pdb
import pickle

if __name__ == "__main__":
	parser = argparse.ArgumentParser('Multi model text classification classifer')
	parser.add_argument('-m', help='model file', type=str, required=True)
	parser.add_argument('-i', help='input file', type=str, required=True)
	parser.add_argument('-o', help='output file', type=str, required=True)
	args = parser.parse_args()
	print('Args:', args)

	with open(args.m, 'rb') as fin:
		model = pickle.load(fin)
		model.test(args.i, args.o)

	print('OK')
