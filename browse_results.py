import shelve
from collections import defaultdict
import sys
import os
import numpy as np
import pandas as pd
import time
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('exp_dir', type = str)
parser.add_argument('-cw', '--col-width', type=int, default=100)
args = parser.parse_args()
print(args)

print('exp_dir=', args.exp_dir)

files = sorted(list(map(lambda x: x[:-4], glob.glob(os.path.join(args.exp_dir, '*.dat')))))


results = defaultdict(list)
ks = [1, 2, 4, 8, 10, 20, 30, 50]
columns=[
        'epoch',
         *['R@{}'.format(i) for i in ks],
        ]

last_modified = None

for p in files:
    try:
        db = shelve.open(p)
        log_path = p + '.log'
        assert os.path.exists(log_path), log_path
        last_modified = (time.time() - os.path.getmtime(p + '.log')) / 60
    except:
        print('Failed to open', p)
    try:
        p = os.path.basename(p)
        cur_results_t = np.array([(epoch, *d['score']['recall'])
                             for (epoch, d) in db['metrics'].items()])
        cur_results = np.zeros((cur_results_t.shape[0], 1 + len(ks)), dtype=float)
        cur_results[:, :] = np.nan
        cur_results[:, :2] = cur_results_t[:, :2]
        # TODO: maybe rename args to config
        if db['config']['dataset_selected'] == 'inshop':
           cur_results[:, 5:] = cur_results_t[:, 2:]
        else:
           cur_results[:, 2:5] = cur_results_t[:, 2:]

    except Exception as e:
        print(p, e)
        print(db['config'])

    idx_max_recall = cur_results[:, 1].argmax()
    best_epoch_results = cur_results[idx_max_recall]
    max_epoch = cur_results[:, 0].max()
    best_epoch_results = best_epoch_results.tolist()
    best_epoch_results[0] = '{:02}/{:02}'.format(int(best_epoch_results[0]), int(max_epoch))
    assert len(best_epoch_results) == len(columns)

    for i, col_name in enumerate(columns):
        results[col_name].append(best_epoch_results[i])

    # if the file was last modified < 10 minute ago; than print Running status
    if last_modified is None:
        results['S'].append('?')
    elif last_modified > 10:
        results['S'].append('-')
    else:
        results['S'].append('[R]')


df = pd.DataFrame(index=list(map(os.path.basename, files)),
                  data=results)

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', args.col_width)
pd.set_option('display.width', 1000000)
df.sort_values(by=['R@1'], inplace=True)
print(df)
