import os
import pandas as pd

from .data_transform import Ion2Vector
from .bucket_utils import merge_buckets


def load_plabel(file_paths, config, nce, instrument):
    ion2vec = Ion2Vector(conf=config, prev=1, next=1)
    buckets = {}
    count = 0
    for filename in file_paths:
        count += 1
        print("%dth plabel" % count, end="\r")
        _buckets = ion2vec.Featurize_buckets(filename, nce, instrument)
        buckets = merge_buckets(buckets, _buckets)
    return buckets


def load_from_folder(dataset_folder, config, nce, instrument='QE'):
    if os.path.isdir(dataset_folder):
        print("Loading %s .." % dataset_folder)
        filenames = []
        for input_file in os.listdir(dataset_folder):
            if input_file.endswith(".plabel"):
                filenames.append(os.path.join(dataset_folder, input_file))
        return load_plabel(filenames, config, nce, instrument)


def load_files_as_buckets(filenames, config, nce, instrument='QE'):
    print("Loading data from files...")
    return load_plabel(filenames, config, nce, instrument)


def load_peptide_file_as_buckets(file_path, config, nce, instrument='QE'):
    """
    The input file format:
        'peptide	modification	charge'
    """
    input_df = pd.read_csv(file_path, sep='\t', low_memory=False)
    input_df = input_df.fillna('')
    ion2vec = Ion2Vector(conf=config, prev=1, next=1)
    buckets = ion2vec.Featurize_buckets_predict(input_df, nce, instrument)
    return buckets
