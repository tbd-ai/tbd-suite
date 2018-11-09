import sys
sys.path.append('../')
import json
import time
import numpy as np

### Import torch ###
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import torch.nn.functional as F

### Import Data Utils ###
from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from params import cuda

def eval_model(model, test_loader, decoder):
"""
Model evaluation -- used during training.
"""
        start_iter = 0
        total_cer, total_wer = 0, 0
        word_count, char_count = 0, 0
        model.eval()
        # For each batch in the test_loader, make a prediction and calculate the WER CER
        for i, (data) in enumerate(test_loader):
            inputs, targets, input_percentages, target_sizes = data
            inputs = Variable(inputs, volatile=True)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            # Decode the ouput to actual strings and compare to label
            # Get the LEV score and the word, char count
            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            for x in range(len(target_strings)):
                total_wer += decoder.wer(decoded_output[x], target_strings[x])
                total_cer += decoder.cer(decoded_output[x], target_strings[x]) 
                word_count += len(target_strings[x].split())
                char_count += len(target_strings[x]) 

            if cuda:
                torch.cuda.synchronize()
            del out
        
        # WER, CER
        wer = total_wer / float(word_count)
        cer = total_cer / float(char_count)
        wer *= 100
        cer *= 100

        return wer, cer

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.array = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.array.append(val)

def eval_model_verbose(model, test_loader, decoder, cuda, n_trials=-1):
"""
Model evaluation -- used during inference.
"""
        start_iter = 0
        total_cer, total_wer = 0, 0
        word_count, char_count = 0, 0
        model.eval()
        batch_time = AverageMeter()
        # We allow the user to specify how many batches (trials) to run
        trials_ran = min(n_trials if n_trials!=-1 else len(test_loader), len(test_loader))
        # For each batch in the test_loader, make a prediction and calculate the WER CER
        for i, (data) in enumerate(test_loader):
            if i < n_trials or n_trials == -1:
                end = time.time()
                inputs, targets, input_percentages, target_sizes = data
                inputs = Variable(inputs, volatile=False)
                
                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                if cuda:
                    inputs = inputs.cuda()
                out = model(inputs)
                out = out.transpose(0, 1)  # TxNxH
                seq_length = out.size(0)
                sizes = input_percentages.mul_(int(seq_length)).int()
                
                # Decode the ouput to actual strings and compare to label
                # Get the LEV score and the word, char count
                decoded_output = decoder.decode(out.data, sizes)
                target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
                for x in range(len(target_strings)):
                    total_wer += decoder.wer(decoded_output[x], target_strings[x])
                    total_cer += decoder.cer(decoded_output[x], target_strings[x]) 
                    word_count += len(target_strings[x].split())
                    char_count += len(target_strings[x])
                    
                # Measure elapsed batch time (time per trial)
                batch_time.update(time.time() - end)
         
                print('[{0}/{1}]\t'
                      'Unorm batch time {batch_time.val:.4f} ({batch_time.avg:.3f})'
                      '50%|99% {2:.4f} | {3:.4f}\t'.format(
                      (i + 1), trials_ran, np.percentile(batch_time.array, 50),
                      np.percentile(batch_time.array, 99), batch_time=batch_time))

                if cuda:
                    torch.cuda.synchronize()
                del out
            else:
                break
                
        # WER, CER
        wer = total_wer  / float(word_count)
        cer = total_cer / float(char_count)
        wer *= 100
        cer *= 100

        return wer, cer, batch_time
