import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
args = parser.parse_args()
TYPES = ['rgb', 'depth']

seqs = os.listdir(args.data_dir)
seqs.sort()
for seq in seqs:
    print(f'Fixing {seq}...')
    for type in TYPES:
        dir = os.path.join(args.data_dir, seq, type)
        files = os.listdir(dir)
        files.sort()
        for f in files:
            old_name = os.path.join(dir, f)
            prefix, suffix = f.rsplit('.', 1)
            try:
                id = int(prefix)
                new_name = os.path.join(dir, f'{id:05d}.{suffix}')
                os.rename(old_name, new_name)
            except:
                print(f'{old_name} cannot be converted. Skip.')
