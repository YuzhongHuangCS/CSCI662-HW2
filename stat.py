import sys
import pdb
import matplotlib.pyplot as plt

import numpy as np
filename = sys.argv[1]
lens = []
for line in open(filename, encoding='utf-8'):
	text, truth = line.rstrip().split('\t')
	if len(text) == 0:
		continue

	lens.append(len(text.split()))

lens = sorted(lens)
print('#len', len(lens))
print('Max len', max(lens))
#print('50%', np.quantile(lens, 0.5))
print('95%', np.quantile(lens, 0.9))

#pdb.set_trace()
'''
plt.hist(lens, normed=True, cumulative=True, label='CDF', histtype='step')
plt.title(filename)
plt.show()
pdb.set_trace()
print('OK')
'''
