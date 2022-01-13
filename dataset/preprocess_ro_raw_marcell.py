""" !Warning, rudimentary script, everything is hardcoded.

Several ways of building a corpus:

1. concatenate all files and iterate line by line
2. concatenate all files and iterate 512 by 512
3. split each file into segments of 512, replace newlines and dump each chunk into a new line (iterate line by line)
4. sentence split all files and iterate line by line

Given the structure of the tests, the best approach might be to go line by line on the concatenation of all files. Each line contains usually a paragraph. Sentence splitting throws errors, splits by comma, is strange.

Way:
Make a corpus cleaning function that removes empty lines and "..." and splits single lines into multiple lines of 128
To train bert efficiently, it's best to prepare the MLM data beforehand. 
1. split into train, test clean, then merge the articles into fixed number of shards.
2. For this type of corpus, it's safe to preserve a max sentence length of 128 tokens.
"""
import os
import random
from tqdm import tqdm

data_dir = './leg/data'
all_files = [os.path.join(data_dir, fis) for fis in tqdm(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, fis))]
print(len(all_files))
random.shuffle(all_files)

split = 0.05
test_size = int(split * len(all_files))
train = all_files[:-test_size]
test = all_files[-test_size:]

print(len(all_files))
print(len(test))
print(len(train))

# this ok
def get_text_clean_split(infile, block_size=128):
    with open(infile, 'r', encoding='utf-8', errors='replace') as fin:
        lines = [line.strip() for line in fin.readlines() if line.strip() and line.strip() != '...']
    newlines = []
    for line in lines:
        wds = line.split(' ')
        if len(wds) > block_size:
            newlines.extend([' '.join(wds[idx:idx+block_size]) for idx in range(0, len(wds), block_size)])
        else:
            newlines.append(line)
    return '\n'.join(newlines)
        

def make_single_file_and_clean(files, out_name, batch=1000):
    with open(out_name, 'w', encoding='utf-8') as fout:
        for idx in tqdm(range(0, len(files), batch)):
            txt_batch = ' '.join([get_text_clean_split(fis) for fis in files[idx:idx+batch]])
            fout.write(txt_batch)


make_single_file_and_clean(test, './leg/test_clean.txt')
make_single_file_and_clean(train, './leg/train_clean.txt')


import os
from itertools import islice
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def single_file_to_shards(fis_in, out_dir, file_prefix, num_shards):
    make_dir(out_dir)
    with open(fis_in, 'r', encoding='utf-8') as fin:
        nr_lines = len(fin.readlines())
    shard_size = int(nr_lines / num_shards)
    with open(fis_in, 'r', encoding='utf-8') as fin:
        for idx, n_lines in enumerate(tqdm(iter(lambda: tuple(islice(fin, shard_size)), ()))):
            out_name = os.path.join(out_dir, file_prefix + str(idx))
            with open(out_name, 'w', encoding='utf-8') as fout:
                fout.write(''.join(n_lines))


single_file_to_shards('./leg/test_clean.txt', './leg/shards', 'testing', 128)
single_file_to_shards('./leg/train_clean.txt', './leg/shards', 'training', 256)