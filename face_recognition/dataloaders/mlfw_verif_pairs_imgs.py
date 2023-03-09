from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path


# BERNARDO
class MLFW_Verif_Pairs_Images:
    
    def load_pairs_samples_protocol_from_file(self, protocol_file_path='pairs.txt', dataset_path='MLFW/aligned', file_ext='.jpg'):
        pos_pair_label = '1'
        neg_pair_label = '0'
        
        with open(protocol_file_path, 'r') as fp:
            all_lines = [line.strip('\n') for line in fp.readlines()]
            # print('all_lines:', all_lines)

            all_pairs = [ None ] * len(all_lines)
            for i, line in enumerate(all_lines):
                sample0, sample1, pair_label = line.split('\t')    # positive=1, negative=0
                path_sample0 = dataset_path + '/' + sample0
                path_sample1 = dataset_path + '/' + sample1
                all_pairs[i] = (path_sample0, path_sample1, pair_label)
                # print(f'mlfw_verif_pairs_imgs - load_pairs_samples_protocol_from_file - all_pairs[{i}]: {all_pairs[i]}')

            return all_pairs, pos_pair_label, neg_pair_label



if __name__ == '__main__':
    pass