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
parser.add_argument('--manifest_stats', default="libri_test_manifest.csv_stats", help='CSV containing audio file name and its duration')
parser.add_argument('--manifest', default="libri_test_manifest.csv_scram_rep", help='CSV containing the results of the inferenc runs')
parser.add_argument('--results', default="cpu_flash_storage_scram_rep.csv", help='CSV containing the results of the inferenc runs')

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
    
    # Load the csv files
    manifest_file = open(osp.join(pwd,args.manifest))
    manifest = csv.reader(manifest_file,delimiter=',')
    manifest_len = len(list(manifest))
    
    manifest_stats_file = open(osp.join(pwd,args.manifest_stats))
    manifest_stats = csv.reader(manifest_stats_file,delimiter=',')
    manifest_stats_len = len(list(manifest_stats))
    
    results_file = open(osp.join(pwd,args.results))
    results = csv.reader(results_file,delimiter=',')
    results_len = len(list(results))
    
    def reset(f): {f.seek(0)}
    reset(manifest_file)
    reset(manifest_stats_file)
    reset(results_file)
    
    # Using the manifest stats file, build a look up form audio file name to duration and a unique sample idx
    audio_stats = dict()
    reset(manifest_stats_file) 
    offset = 0
    for idx, row in enumerate(manifest_stats):
        # row = <audio file name>, <clip duration [sec]>, <running average>
        audio_stats[osp.basename(row[0])] = (idx, float(row[1]))
        offset = idx + 1
        
    # Make correspondence between manifest and result map from run number to audio filename
    # For some reaon... Maybe manifest_stats_file didn't include all the audio files..
    reset(manifest_file) 
    run_num_to_audioname = []
    for idx, row in enumerate(manifest):
        # row = <audio file name>, <transcript file name>
        audioname = osp.basename(row[0])
        if not audioname in audio_stats:
            audio_stats[audioname] = (offset + idx, -1) 
        run_num_to_audioname.append(audioname)
    
    # Preprocess the resuts so that only the runtimes remain in a 1d array
    start = np.inf
    runtime_res = []
    reset(results_file)
    for i, row in enumerate(results):
        if row[0] == 'data':
            start = i + 1
        runtime_res.append(row[0])
        # only convert the entries we know are runtime data
        if i >= start:
            runtime_res[-1] = float(runtime_res[-1])
    assert (start > 0 and start < results_len-1), "data tag for results_file not found in valid position"
    runtime_res = runtime_res[start:]

    # 1. Plot the distribution of each sample's runtimes, plot the warmups runs in RED
    # 1b. Compute mean sample runtime and standard deviation (excluding warmups).
    IDX = 0; DUR = 1; TIME =2;
    num_warmups = 50 
    data = dict()
    fig1 = plt.figure(1)
    ax = plt.subplot(1, 1, 1)
    for run_num, audioname in enumerate(run_num_to_audioname):
        color = 'r' 
        stat = list(audio_stats[audioname])
        stat.append(runtime_res[run_num])
        if run_num > num_warmups:
            color = 'b'
            # Update the data dictionary to compute error bars and print out a detailed summary
            idx = stat[IDX]
            if not idx in data:
                data[idx] = {'series': [stat[TIME]],'n': 1, 'mean': stat[TIME], 'stddev': 0}
            else:
                prev = data[idx]                
                data[idx]['series'].append(stat[TIME])
                data[idx]['n'] += 1
                data[idx]['mean'] = sum(data[idx]['series']) / data[idx]['n']
                data[idx]['stddev'] = np.std(data[idx]['series'])
        ax.plot(stat[IDX],stat[TIME],marker='o',c=color)
#         plt_show_no_blk(t=0.0001)
        sys.stdout.write("\r[{}/{}]         ".format(run_num, len(run_num_to_audioname)))
        sys.stdout.flush()
    print('Save data')
    data_output = osp.join(pwd, "sample_runtime.csv")
    make_file(data_output)
    write_line(data_output, "idx,n,mean,stddev,trials\n")
    for idx in data:
        series_str = ""
        for trial in data[idx]['series']:
            series_str = series_str + str(trial) + ","
        write_line(data_output, "{},{},{},{},{}\n".format(idx,data[idx]['n'],
                                                        data[idx]['mean'],
                                                        data[idx]['stddev'],
                                                        series_str))
        sys.stdout.write("\r[{}/{}]         ".format(idx, len(data)))
        sys.stdout.flush()
    print('Plot error bars')
    for idx in data:
        ax.errorbar(idx,data[idx]['mean'],yerr=data[idx]['stddev'])
    plt.title('Sactter plot of inference trials and variances of Librispeech Test Clean inputs')
    plt.xlabel('Idx of input')
    plt.ylabel('Latency of inference trials')
    print('Showing plot (takes a while and blocks)')
#     plt.show()
    print('Saving Figures')
    plt.savefig("sample_runtime.jpg")
    print('fin')
    
    # 2. Remove some x warmup runs then plot the CDF
    print('Generating histogram')
    hist, bin_edges = np.histogram(runtime_res,bins=70)
    print('Generating cdf')
    cdf = np.cumsum(hist)
    print('Ploting cdf')
    fig2 = plt.figure(2)
    ax2 = plt.subplot(1, 1, 1)
    ax2.plot(bin_edges[1:],cdf)
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('% of samples')
    plt.title('% of batch size 1 inputs from Librispeech Test Clean satisfying a latency bound')
    plt.xticks(bin_edges,rotation=90)
    plt.yticks(cdf[::10], np.round(cdf/cdf[-1],2)[::10])
    plt.axhline(y=0.99*cdf[-1],xmin=0,xmax=bin_edges[-1],c='k')
    plt.axvline(x=bin_edges[find_nearest_idx(cdf/cdf[-1], 0.99)],ymin=0, ymax=1,c='k')
    print('Showing plot')
    plt.show()
    print('fin')
    
    # 3. Plot the distribution of audio durations
    DUR = 1
    print('Extracing audio durations')
    audio_dur = [audio_stats[key][DUR] for key in audio_stats]
    print(max(audio_dur))
    print('Cleaning invalid durations')
    while -1 in audio_dur:
        audio_dur.remove(-1)
    print('Generating histogram')
    hist, bin_edges = np.histogram(audio_dur,bins=70)
    print('Plotting histogram')
    fig3 = plt.figure(3)
    ax3 = plt.subplot(1,1,1)
    ax3.plot(bin_edges[1:],hist)
    plt.xticks(bin_edges,rotation=90)
    plt.title('Audio clip duration Histogram for Librispeech Test Clean')
    plt.xlabel('Duration of audio clip [sec]')
    plt.ylabel('Count')
    print('Showing plot')
    plt.show()
    print('fin')
    
    