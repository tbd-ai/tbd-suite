import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

# <EcoSys> Added output_directory argument.
def flush(output_directory):
	# </EcoSys>
	prints = []

	for name, vals in _since_last_flush.items():
		prints.append("{}\t{}".format(name, np.mean(vals.values())))
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		# <EcoSys> Added output_directory to path.
		plt.savefig(output_directory + name.replace(' ', '_')+'.jpg')
		# </EcoSys>
	
	print "iter {}\t{}".format(_iter[0], "\t".join(prints))
	_since_last_flush.clear()
	
	# <EcoSys> Added output_directory to path.
	with open(output_directory + 'log.pkl', 'wb') as f:
		# </EcoSys>
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

