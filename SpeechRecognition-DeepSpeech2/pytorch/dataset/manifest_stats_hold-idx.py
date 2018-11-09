import os.path as osp
import os
import csv
import argparse
import sys
import sox

example_wavfile = '/scratch/jacob/training/speech_recognition/LibriSpeech_dataset/test/wav/2414-128291-0020.wav'
example_txtfile = '/scratch/jacob/training/speech_recognition/LibriSpeech_dataset/test/txt/2414-128291-0020.txt'

def make_folder(filename):
    temp = osp.dirname(filename)
    if not osp.exists(temp):
        print("Making folder at: {}".format(temp))
        os.makedirs(temp)

def make_file(filename,data=None):
    f = open(filename,"w+")
    f.close()
    if data:
        write_line(filename,data)

def write_line(filename,msg):
    f = open(filename,"a")
    f.write(msg)
    f.close()

def format_entry(entry, root):
    base = osp.basename(entry[0])
    folder = osp.basename(osp.dirname(entry[0]))
    base = base.split('.')[0]+".txt"
    new_file = osp.join(root,folder,base)
    new_entry = entry[2].upper()
    return (new_file, new_entry)

def make_manifest(inputfile, root, idx):
    if idx == -1:
        idx = ""
    else:
        idx = '_held'
    base = osp.basename(inputfile)
    base = base + idx
    manifest_file = osp.join(root, base)
    make_folder(manifest_file)
    make_file(manifest_file)
    return manifest_file

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='some manifest file with the first col containing the wav file path')
    parser.add_argument('--hold_idx', default=-1, type=int)
    parser.add_argument('--stats', dest='stats', action='store_true')
    args = parser.parse_args()
    root = os.getcwd()		# the root is the current working directory
    filepath = osp.join(os.getcwd(),args.file)
    print("\n\nOpening: {}".format(filepath))
    print("Root: {}".format(root))
    if args.stats:
        manifest_file = filepath + "_stats"
        make_folder(manifest_file)
        make_file(manifest_file)
        audio_dur = AverageMeter()
    else: 
        manifest_file = make_manifest(filepath, root, args.hold_idx)
    print("Manifest made: {}".format(manifest_file))
    f = open(filepath)
    summary = csv.reader(f,delimiter=',')
    tot = 0
    hold_file = ""
    hold_entry = ""
    for i, row in enumerate(summary):
        tot += 1;
        if args.hold_idx == i:
            #(hold_file, hold_entry) = format_entry(row, root)
            hold_file = row[0]
            hold_entry = row[1]
    cur = 0
    f.seek(0)
    new_file = hold_entry
    for row in summary:
        if cur == 0:
            cur += 1
            continue
        if not args.stats:
            if args.hold_idx != -1:
                write_line(manifest_file, hold_file+","+hold_entry+"\n")
            else:
                exit(1)
                (new_file, new_entry) = format_entry(row, root)
                make_folder(new_file)
                make_file(new_file, new_entry)
        else:
            seconds = sox.file_info.duration(row[0])
            audio_dur.update(seconds)
            new_file = "{},{}".format(seconds, audio_dur.avg)
            write_line(manifest_file, row[0]+","+new_file+"\n")
        sys.stdout.write("\r[{}/{}] {}         ".format(cur,tot,new_file))
        sys.stdout.flush()
        cur += 1
    sys.stdout.write("\r[{}/{}] {}         ".format(cur,tot,new_file))
    sys.stdout.flush()
    print("\n")

