#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

# import numpy as np

import multiprocessing as mp
import multiprocessing.pool
from sklearn.cluster import KMeans
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

def my_fun(params):
    idx = params[0]
    val = params[1]
    print("::Random {0} -> {1}".format(idx, val))
    tmat = np.random.random( (220, 220) )
    print("::Eigen {0} -> {1}".format(idx, val))
    tret = np.linalg.eig(tmat)
    print("::Mean {0} -> {1}".format(idx, val))
    tret = np.mean(tret[1])
    # tdat = np.random.random((10000,100))
    # print("::KMeans {0} -> {1}".format(idx, val))
    # km = KMeans(n_clusters=100, n_jobs=1).fit(tdat)
    print('{1} : ret = {0}'.format(tret, idx))
    return 0

def map_fun(data):
    return np.sum(data)

if __name__ == '__main__':
    # threadPoolExecutor = ThreadPoolExecutor(max_workers=4)
    p0 = mp.Pool(processes=6)
    ress = p0.map(map_fun, [np.random.random(10) for xx in range(100)])
    #
    tmp = []
    pool = mp.pool.ThreadPool(processes=4)
    print ('----- RUN THREADS -----')
    for iidx in range(16):
        # pool.apply_async(my_fun, [(iidx, float(iidx)/2.)])
        # tmp.append(threadPoolExecutor.submit(my_fun, [iidx, float(iidx)/2.]))
        t = threading.Thread(target=my_fun, args = [(iidx, float(iidx)/2.)])
        tmp.append(t)
        t.start()
    pool.close()
    pool.join()
    print ('----- WAIT THREADS -----')
    # for tt in tmp:
    #     tt.join()

