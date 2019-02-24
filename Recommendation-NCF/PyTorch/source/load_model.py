import os
from argparse import ArgumentParser
import torch
from neumf import NeuMF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)

def parse_args():
    parser = ArgumentParser(description="Load a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('--path', type=str, help='Path to pretrained model')
    return parser.parse_args()

args = parse_args()

print('Loading data')
train_dataset = CFTrainDataset(
    os.path.join('ml-20m', TRAIN_RATINGS_FILENAME), 4)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=2048, shuffle=True,
    num_workers=0, pin_memory=True)

test_ratings = load_test_ratings(os.path.join('ml-20m', TEST_RATINGS_FILENAME))  # noqa: E501
test_negs = load_test_negs(os.path.join('ml-20m', TEST_NEG_FILENAME))
nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items

# Create model
layers=[256, 256, 128, 64]
model = NeuMF(nb_users, nb_items,
                  mf_dim=64, mf_reg=0.,
                  mlp_layer_sizes=layers,
                  mlp_layer_regs=[0. for i in layers])
model.load_state_dict(torch.load(args.path))
model = model.cuda

print(model)

# Your code here
