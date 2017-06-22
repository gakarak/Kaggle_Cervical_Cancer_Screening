#!/usr/bin/python

import os
import sys
import h5py
import glob

if __name__=='__main__':
	wdir='.'
	if len(sys.argv)>2:
		wdir = sys.argv[1]
	else:
		print ('Usage: {0} [/path/to/dir-with-models/*.h5]')
	if not os.path.isdir(wdir):
		raise Exception('*** ERROR *** Cant find directory [{0}]'.format(wdir))
	lstH5 = glob.glob('{0}/*.h5'.format(wdir))
	numH5 = len(lstH5)
	print(':: Working in directory [{0}]:'.format(wdir))
	for ipath, path in enumerate(lstH5):
		print('\tprocessing H5: [{0}]'.format(path))
		with h5py.File(path, 'a') as f:
			if 'optimizer_weights' in f.keys():
				del f['optimizer_weights']
		# f = h5py.File(path, 'r+')
		# del f['optimizer_weights']
		# f.close()
	print('... done')
