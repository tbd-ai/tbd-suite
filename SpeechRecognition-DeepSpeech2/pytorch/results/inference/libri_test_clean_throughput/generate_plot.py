import os
import sys
import csv
import time
import argparse
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from astropy.units import ymin

parser = argparse.ArgumentParser(description='DeepSpeech inference graph plotter')

def make_file(filename,data=None):
    f = open(filename,"w+")
    f.close()
    if data:
        write_line(filename,data)

def write_line(filename,msg):
    f = open(filename,"a")
    f.write(msg)
    f.close()
    
def plt_show_no_blk(t=0.01):
    plt.ion()
    plt.show()
    plt.pause(t)
    
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":
    args = parser.parse_args()
    pwd = os.getcwd()
    
    def reset(f): {f.seek(0)}
    
    def load_results(f):
        pwd = os.getcwd()
        results_file = open(osp.join(pwd,f))
        results = csv.reader(results_file,delimiter=',')
        results_len = len(list(results))
        return results_file, results, results_len
    
    def preproc_results(results_tuple):
        # Preprocess the resuts so that only the runtimes remain in a 1d array
        results_file, results, results_len = results_tuple
        start = np.inf
        runtime_res = []
        meta = []
        reset(results_file)
        for i, row in enumerate(results):
            if row[0] == 'data':
                start = i + 1
            runtime_res.append(row[0])
            # only convert the entries we know are runtime data
            if i >= start:
                runtime_res[-1] = float(runtime_res[-1])
            else:
                meta.append(row)
        assert (start > 0 and start < results_len-1), "data tag for results_file not found in valid position"
        return meta, runtime_res[start:]  # meta and data
    
    folders = ['2s','5s','5s_rerun']
    colors = ['k','b','g','r']
    batch_sizes = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    DUR = 0; BS = 4; P50 = 7; P99 = 8;
    num_warmups = 0 
    
    
    data = dict()
    fig1 = plt.figure(1)
    ax = plt.subplot(1, 1, 1)
    for folder, color in zip(folders,colors):
        for batch_size in batch_sizes:
            res_path = osp.join(pwd,folder,"{}.csv".format(batch_size))
            meta, _ = preproc_results(load_results(res_path))
            ax.plot(float(meta[P99][1])*batch_size,batch_size*float(meta[DUR][1]),marker="o",c=color)
            sys.stdout.write("\r[{}/{}]         ".format(folder, batch_size))
            sys.stdout.flush()

    plt.title('Throughput vs Latency of Librispeech Test Clean inputs')
    plt.xlabel('99%-tile latency for one batch [sec]')
    plt.ylabel('Throughput, total audio duration of batch [sec]')
    plt.legend(folders,loc='best')
    print('Showing plot')
    plt.show()
#     print('Saving Figures')
#     plt.savefig("sample_runtime.jpg")
    print('fin')