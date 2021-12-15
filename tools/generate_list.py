import os
from os.path import join
import argparse
import random

TYPE = 'rgb'

parser = argparse.ArgumentParser()

parser.add_argument('data_dir')
parser.add_argument('--split', required=True)
args = parser.parse_args()

data_dir = os.path.join(args.data_dir)
seqs = list(
    filter(lambda x: os.path.isdir(os.path.join(data_dir, x)),
           os.listdir(data_dir)))

train_txt = []
val_txt = []

seqs = list(seqs)
seqs.sort()
for idx, seq in enumerate(seqs):
    # Skip unvalid directories
    if 'det' in seq or '0345' in seq:
        continue
    files = os.listdir(join(data_dir, seq, TYPE))
    files.sort()
    name_mapping = lambda x : f'{seq}/{TYPE} {x}\n'

    if '000' in seq:
        val_txt += list(map(name_mapping, files[1:-1]))
    else:
        train_txt += list(map(name_mapping, files[1:-1]))

    # num_train = int(len(files) * 0.8)
    # train_txt += list(map(name_mapping, files[1:num_train]))
    # val_txt += list(map(name_mapping, files[num_train:-1]))

    # if idx == len(seqs) - 1:
    #     val_txt += list(map(name_mapping, files[1:-1]))
    # else:
    #     train_txt += list(map(name_mapping, files[1:-1]))

file_dir = os.path.abspath(__file__).rsplit('/', 2)[0]

os.makedirs(f'splits/{args.split}', exist_ok=True)
with open(join(file_dir, f'splits/{args.split}/train_files.txt'), 'w') as f:
    random.shuffle(train_txt)
    f.writelines(train_txt)
with open(join(file_dir, f'splits/{args.split}/val_files.txt'), 'w') as f:
    random.shuffle(val_txt)
    f.writelines(val_txt)
