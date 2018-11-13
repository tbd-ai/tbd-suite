import argparse
import errno
import json
import os
import os.path as osp
import time

import sys

import numpy as np

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

### Import Data Utils ###
sys.path.append('../')

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

import params

from eval_model import  eval_model_verbose

###########################################################
# Comand line arguments, handled by params except seed    #
###########################################################
parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')

parser.add_argument('--seed', default=0xdeadbeef, type=int, help='Random Seed')

parser.add_argument('--acc', default=23.0, type=float, help='Target WER')

parser.add_argument('--start_epoch', default=-1, type=int, help='Number of epochs at which to start from')

parser.add_argument('--use_set', default="libri", help='ov = OpenVoice test set, libri = Librispeech val set')

parser.add_argument('--cpu', default=0, type=int)

parser.add_argument('--hold_idx', default=-1, type=int, help='input idx to hold the test dataset at')

parser.add_argument('--hold_sec', default=1, type=float)

parser.add_argument('--batch_size_val', default=-1, type=int)

parser.add_argument('--n_trials', default=-1, type=int, help='limit the number of trial ran, useful when holding idx')


def to_np(x):
    return x.data.cpu().numpy()

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

def make_file(filename,data=None):
    f = open(filename,"w+")
    f.close()
    if data:
        write_line(filename,data)

def write_line(filename,msg):
    f = open(filename,"a")
    f.write(msg)
    f.close()

def main():
    args = parser.parse_args()

    params.cuda = not bool(args.cpu)
    print("Use cuda: {}".format(params.cuda))

    torch.manual_seed(args.seed)
    if params.cuda:
      torch.cuda.manual_seed_all(args.seed)

    if params.rnn_type == 'gru' and params.rnn_act_type != 'tanh':
      print("ERROR: GRU does not currently support activations other than tanh")
      sys.exit()

    if params.rnn_type == 'rnn' and params.rnn_act_type != 'relu':
      print("ERROR: We should be using ReLU RNNs")
      sys.exit()

    print("=======================================================")
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

    save_folder = args.save_folder

    loss_results, cer_results, wer_results = torch.Tensor(params.epochs), torch.Tensor(params.epochs), torch.Tensor(params.epochs)
    best_wer = None
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    criterion = CTCLoss()

    with open(params.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    audio_conf = dict(sample_rate=params.sample_rate,
                      window_size=params.window_size,
                      window_stride=params.window_stride,
                      window=params.window,
                      noise_dir=params.noise_dir,
                      noise_prob=params.noise_prob,
                      noise_levels=(params.noise_min, params.noise_max))

    if args.use_set == 'libri':
        testing_manifest = params.val_manifest + ("_held{}".format(args.hold_idx) if args.hold_idx >=0 else "")
    else:
        testing_manifest = params.test_manifest

    if args.batch_size_val > 0:
        params.batch_size_val = args.batch_size_val

    print("Testing on: {}".format(testing_manifest))
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.val_manifest, labels=labels,
                                       normalize=True, augment=params.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=testing_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_loader = AudioDataLoader(train_dataset, batch_size=params.batch_size,
                                   num_workers=1)
    test_loader = AudioDataLoader(test_dataset, batch_size=params.batch_size_val,
                                  num_workers=1)

    rnn_type = params.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    model = DeepSpeech(rnn_hidden_size = params.hidden_size,
                       nb_layers       = params.hidden_layers,
                       labels          = labels,
                       rnn_type        = supported_rnns[rnn_type],
                       audio_conf      = audio_conf,
                       bidirectional   = False,
                       rnn_activation  = params.rnn_act_type,
                       bias            = params.bias)

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=params.lr,
                                momentum=params.momentum, nesterov=True,
                                weight_decay = params.l2)
    decoder = GreedyDecoder(labels)

    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)

        if params.cuda:
            package = torch.load(args.continue_from)
        else:
            package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1)) - 1  # Python index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))

        if args.start_epoch != -1:
          start_epoch = args.start_epoch

        avg_loss = 0
        start_epoch = 0
        start_iter = 0
        avg_training_loss = 0
        epoch = 1
    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0
        avg_training_loss = 0
    if params.cuda:
        model         = torch.nn.DataParallel(model).cuda()
        # model         = torch.nn.parallel.DistributedDataParallel(model).cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ctc_time = AverageMeter()

    for epoch in range(start_epoch, params.epochs):

        #################################################################################################################
        #                    The test script only really cares about this section.
        #################################################################################################################
        model.eval()

        wer, cer, trials = eval_model_verbose( model, test_loader, decoder, params.cuda, args.n_trials)
        root = os.getcwd()
        outfile = osp.join(root, "inference_bs{}_i{}_gpu{}.csv".format(params.batch_size_val, args.hold_idx, params.cuda))
        print("Exporting inference to: {}".format(outfile))
        make_file(outfile)
        write_line(outfile, "batch times pre normalized by hold_sec =,{}\n".format(args.hold_sec))
        write_line(outfile, "wer, {}\n".format(wer))
        write_line(outfile, "cer, {}\n".format(cer))
        write_line(outfile, "bs, {}\n".format(params.batch_size_val))
        write_line(outfile, "hold_idx, {}\n".format(args.hold_idx))
        write_line(outfile, "cuda, {}\n".format(params.cuda))
        write_line(outfile, "avg batch time, {}\n".format(trials.avg/args.hold_sec))
        percentile_50 = np.percentile(trials.array,50)/params.batch_size_val/args.hold_sec
        write_line(outfile, "50%-tile latency, {}\n".format(percentile_50))
        percentile_99 = np.percentile(trials.array,99)/params.batch_size_val/args.hold_sec
        write_line(outfile, "99%-tile latency, {}\n".format(percentile_99))
        write_line(outfile, "through put, {}\n".format(1/percentile_50))
        write_line(outfile, "data\n")
        for trial in trials.array:
            write_line(outfile, "{}\n".format(trial/args.hold_sec))

        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / params.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        break

    print("=======================================================")
    print("***Best WER = ", best_wer)
    for arg in vars(args):
      print("***%s = %s " %  (arg.ljust(25), getattr(args, arg)))
    print("=======================================================")

if __name__ == '__main__':
    main()
