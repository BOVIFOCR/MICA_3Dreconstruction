from __future__ import print_function

import sys
import os
import numpy as np
from glob import glob
from pathlib import Path
import random

import tarfile
import os.path



# BERNARDO
class TreeMS1MV3_3DReconstructedMICA:

    def get_all_sub_folders(self, dir_path='', dir_level=2):
        return sorted(glob(dir_path + '/*'*dir_level))

    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)

    def get_all_pointclouds_paths(self, dir_path, dir_level=2, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path, dir_level)
        all_pc_paths = []
        all_pc_subjects = []
        # print('all_sub_folders:', all_sub_folders)
        # print('len(all_sub_folders):', len(all_sub_folders))
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            # print('pc_paths:', pc_paths)
            # assert len(pc_paths) > 0
            if not len(pc_paths) > 0:
                raise Exception(f'Error, no file found by pattern \'*{pc_ext}\' in folder \'{sub_folder_pointcloud}\'')
            pc_subjects = [pc_path.split('/')[-3] for pc_path in pc_paths]
            # print('pc_subjects:', pc_subjects)
            assert len(pc_subjects) > 0
            # print('----------------------')
            # raw_input('PAUSED')
            all_pc_paths += pc_paths
            all_pc_subjects += pc_subjects
        
        assert len(all_pc_paths) > 0
        assert len(all_pc_subjects) > 0
        return all_pc_paths, all_pc_subjects


    def count_samples_per_subject(self, pc_paths_list=[''], dir_level=2, pc_ext='.ply'):
        unique_subjects_names = []
        samples_per_subject = []
        indexes_samples = []
        for i, pc_path in enumerate(sorted(pc_paths_list)):
            # print('%d/%d - %s' % (i, len(pc_paths_list), pc_path), end='\r')
            print('%d/%d - \'%s\'' % (i, len(pc_paths_list), pc_path), end='\r')
            data_path = pc_path.split('/')
            if pc_ext in data_path[-1]:
                subject_name = data_path[-(dir_level+1)]
                if not subject_name in unique_subjects_names:
                    samples_per_subject.append(0)
                    unique_subjects_names.append(subject_name)
                    indexes_samples.append([i, i-1])    # i=begin, i-1=end

                samples_per_subject[-1] += 1
                indexes_samples[-1][1] += 1   # increment end index
        assert len(unique_subjects_names) == len(samples_per_subject)
        print()
        return unique_subjects_names, samples_per_subject, indexes_samples

    def get_all_pointclouds_paths_count(self, dir_path, dir_level=2, pc_ext='.ply'):
        print(f'get_all_pointclouds_paths_count() - getting all pointclouds paths from: \'{dir_path}\'')
        all_pc_paths, all_pc_subjects = self.get_all_pointclouds_paths(dir_path, dir_level, pc_ext)
        print(f'get_all_pointclouds_paths_count() - counting samples per subject from: \'{dir_path}\'')
        unique_subjects_names, samples_per_subject, indexes_samples = self.count_samples_per_subject(all_pc_paths, dir_level, pc_ext)
        return all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples

    def filter_paths_by_minimum_samples(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext='.ply', min_samples=2, max_samples=-1):
        filtered_pc_paths = []
        filtered_pc_subjects = []
        filtered_subjects_names = []
        filtered_samples_per_subject = []
        selected_samples_per_subject = [0] * len(unique_subjects_names)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            # print('%d/%d - %s' % (i+1, len(all_pc_paths), pc_path), end='\r')
            print('%d/%d - \'%s\'' % (i+1, len(all_pc_paths), pc_path), end='\r')
            if samples_per_subject[unique_subjects_names.index(pc_subj)] >= min_samples and \
               (max_samples==-1 or selected_samples_per_subject[unique_subjects_names.index(pc_subj)] < max_samples):
                filtered_pc_paths.append(pc_path)
                filtered_pc_subjects.append(pc_subj)
                if not pc_subj in filtered_subjects_names:   # run once per subject
                    filtered_subjects_names.append(pc_subj)
                selected_samples_per_subject[unique_subjects_names.index(pc_subj)] += 1
        # filtered_samples_per_subject.append(samples_per_subject[unique_subjects_names.index(pc_subj)])
        filtered_samples_per_subject = [selected_samples_per_subject[unique_subjects_names.index(pc_subj)] for pc_subj in filtered_subjects_names]
        # print('selected_samples_per_subject:', selected_samples_per_subject)      
        print()
        return filtered_pc_paths, filtered_pc_subjects, filtered_subjects_names, filtered_samples_per_subject

    def load_filter_organize_pointclouds_paths(self, dir_path, dir_level=2, pc_ext='.ply', min_samples=2, max_samples=-1):
        if not os.path.isdir(dir_path): raise Exception(f'Error, dir not found: \'{dir_path}\'')
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples = self.get_all_pointclouds_paths_count(dir_path, dir_level=dir_level, pc_ext=pc_ext)
        print('load_filter_organize_pointclouds_paths() - filtering paths by min and max samples ...')
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext, min_samples, max_samples)
        subjects_with_pc_paths = [()] * len(all_pc_paths)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            subjects_with_pc_paths[i] = (pc_subj, pc_path)
        # print('samples_per_subject:', samples_per_subject)
        return subjects_with_pc_paths, unique_subjects_names, samples_per_subject

    def load_pairs_samples_protocol_from_file(self, protocol_file_path='pairsDevTrain.txt', dataset_path='', file_ext='.ply'):
        pos_pair_label = '1'
        neg_pair_label = '0'
        all_pos_pairs_paths = []
        all_neg_pairs_paths = []

        with open(protocol_file_path, 'r') as fp:
            all_lines = [line.rstrip('\n') for line in fp.readlines()]
            # print('all_lines:', all_lines)
            num_pos_pairs = int(all_lines[0])
            for i in range(1, num_pos_pairs+1):
                pos_pair = all_lines[i].split('\t')   # Aaron_Peirsol	1	2
                # print('pos_pair:', pos_pair)
                subj_name, index1, index2 = pos_pair
                path_sample1 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index1.zfill(4), '*'+file_ext))[0]
                path_sample2 = glob(os.path.join(dataset_path, subj_name, subj_name+'_'+index2.zfill(4), '*'+file_ext))[0]
                # pos_pair = (subj_name, pos_pair_label, path_sample1, path_sample2)
                pos_pair = (pos_pair_label, path_sample1, path_sample2)
                all_pos_pairs_paths.append(pos_pair)
                # print('path_sample1:', path_sample1)
                # print('path_sample2:', path_sample2)
                # print('pos_pair:', pos_pair)

            for i in range(num_pos_pairs+1, len(all_lines)):
                neg_pair = all_lines[i].split('\t')   # AJ_Cook	1	Marsha_Thomason	1
                # print('neg_pair:', neg_pair)
                subj_name1, index1, subj_name2, index2 = neg_pair
                path_sample1 = glob(os.path.join(dataset_path, subj_name1, subj_name1+'_'+index1.zfill(4), '*'+file_ext))[0]
                path_sample2 = glob(os.path.join(dataset_path, subj_name2, subj_name2+'_'+index2.zfill(4), '*'+file_ext))[0]
                neg_pair = (neg_pair_label, path_sample1, path_sample2)
                all_neg_pairs_paths.append(neg_pair)
                # sys.exit(0)
            return all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label


    def make_pairs_global_indexes(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples=True):
        random.seed(440)

        def choose_random_sample(begin, end, amount=1):
            return random.sample(range(begin, end+1), amount)[0]

        def make_random_pair(begin, end, amount=2):
            return random.sample(range(begin, end+1), amount)

        def is_pair_valid(avail_all_pc_paths, idx1, idx2):
            if avail_all_pc_paths[idx1] == True and avail_all_pc_paths[idx2] == True:
                return True
            return False

        def is_pair_duplicate(pair_search, all_pairs):
            assert len(pair_search) == 2
            found_indexes = [idx for idx, pair in enumerate(all_pairs) if pair_search[0] in pair and pair_search[1] in pair]
            if len(found_indexes) > 0:
                return True
            return False

        pos_pairs = [None] * num_pos_pairs
        neg_pairs = [None] * num_neg_pairs
        avail_all_pc_paths = [True] * len(all_pc_paths)

        # Make positive pairs
        rand_subj_idx = random.sample(range(0, len(unique_subjects_names)), len(unique_subjects_names))
        pair_idx = 0
        subj_idx = 0
        while pair_idx < num_pos_pairs:
            if samples_per_subject[rand_subj_idx[subj_idx]] > 1:   # for positive pairs, use only subjects containing 2 or more samples
                begin_subj, end_subj = indexes_samples[rand_subj_idx[subj_idx]]

                one_pos_pair_idx = make_random_pair(begin_subj, end_subj, amount=2)
                while is_pair_duplicate(one_pos_pair_idx, pos_pairs[:pair_idx]):
                    print('Duplicate positive pair found:', one_pos_pair_idx, '    formed pairs:', str(pair_idx)+'/'+str(num_pos_pairs))
                    subj_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]
                    if samples_per_subject[rand_subj_idx[subj_idx]] > 1:
                        begin_subj, end_subj = indexes_samples[rand_subj_idx[subj_idx]]
                        one_pos_pair_idx = make_random_pair(begin_subj, end_subj, amount=2)

                if not reuse_samples:
                    while not is_pair_valid(avail_all_pc_paths, one_pos_pair_idx[0], one_pos_pair_idx[1]):
                        one_pos_pair_idx = make_random_pair(begin_subj, end_subj, amount=2)

                avail_all_pc_paths[one_pos_pair_idx[0]], avail_all_pc_paths[one_pos_pair_idx[1]] = False, False
                pos_pairs[pair_idx] = [unique_subjects_names[rand_subj_idx[subj_idx]], one_pos_pair_idx[0], one_pos_pair_idx[1]]
                # print(str(pair_idx)+'/'+str(num_pos_pairs) + '    subject_name:', unique_subjects_names[rand_subj_idx[subj_idx]], '    samples_per_subject:', samples_per_subject[rand_subj_idx[subj_idx]], '    indexes_samples:', indexes_samples[rand_subj_idx[subj_idx]], '    pos_pairs[pair_idx]:', pos_pairs[pair_idx])
                # raw_input('PAUSED')
                pair_idx += 1
            subj_idx += 1
            if subj_idx >= len(unique_subjects_names):
                subj_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]

        # Make negative pairs
        rand_subj_idx = random.sample(range(0, len(unique_subjects_names)), len(unique_subjects_names))
        pair_idx = 0
        subj1_idx, subj2_idx = 0, 1
        while pair_idx < num_neg_pairs:
            begin_subj1, end_subj1 = indexes_samples[rand_subj_idx[subj1_idx]]
            begin_subj2, end_subj2 = indexes_samples[rand_subj_idx[subj2_idx]]

            one_neg_pair_idx = [choose_random_sample(begin_subj1, end_subj1, amount=1), choose_random_sample(begin_subj2, end_subj2, amount=1)]
            while is_pair_duplicate(one_neg_pair_idx, neg_pairs[:pair_idx]):
                print('Duplicate negative pair found:', one_neg_pair_idx, '    formed pairs:', str(pair_idx)+'/'+str(num_neg_pairs))
                subj1_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]
                subj2_idx = subj1_idx+1
                begin_subj1, end_subj1 = indexes_samples[rand_subj_idx[subj1_idx]]
                begin_subj2, end_subj2 = indexes_samples[rand_subj_idx[subj2_idx]]
                one_neg_pair_idx = [choose_random_sample(begin_subj1, end_subj1, amount=1), choose_random_sample(begin_subj2, end_subj2, amount=1)]

            if not reuse_samples:
                while not is_pair_valid(avail_all_pc_paths, one_neg_pair_idx[0], one_neg_pair_idx[1]):
                    one_neg_pair_idx = [choose_random_sample(begin_subj1, end_subj1, amount=1), choose_random_sample(begin_subj2, end_subj2, amount=1)]

            avail_all_pc_paths[one_neg_pair_idx[0]], avail_all_pc_paths[one_neg_pair_idx[1]] = False, False
            neg_pairs[pair_idx] = [unique_subjects_names[rand_subj_idx[subj1_idx]], one_neg_pair_idx[0], unique_subjects_names[rand_subj_idx[subj2_idx]], one_neg_pair_idx[1]]
            # print('subject_name1:', unique_subjects_names[rand_subj_idx[subj1_idx]], '    samples_per_subject1:', samples_per_subject[rand_subj_idx[subj1_idx]], '    indexes_samples1:', indexes_samples[rand_subj_idx[subj1_idx]])
            # print('subject_name2:', unique_subjects_names[rand_subj_idx[subj2_idx]], '    samples_per_subject2:', samples_per_subject[rand_subj_idx[subj2_idx]], '    indexes_samples2:', indexes_samples[rand_subj_idx[subj2_idx]])
            # print(str(pair_idx)+'/'+str(num_neg_pairs) + '    neg_pairs[pair_idx]:', neg_pairs[pair_idx])            
            # raw_input('PAUSED')
            # print('--------------------------')
            pair_idx += 1
            subj1_idx += 2
            subj2_idx += 2

            if subj2_idx >= len(unique_subjects_names):
                subj1_idx = random.sample(range(0, len(unique_subjects_names)), 1)[0]
                subj2_idx = subj1_idx+1

        return pos_pairs, neg_pairs


    def make_pairs_indexes_lfw_format(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples=True):
        pos_pairs, neg_pairs = self.make_pairs_global_indexes(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)

        for i in range(len(pos_pairs)):    # pos_pairs[i] = [0=subj_name, 1=global_idx1, 2=global_idx2]
            file_name_sample1 = all_pc_paths[pos_pairs[i][1]].split('/')[-2]
            file_name_sample2 = all_pc_paths[pos_pairs[i][2]].split('/')[-2]
            pos_pairs[i][1] = file_name_sample1
            pos_pairs[i][2] = file_name_sample2
            # print('pos_pairs[i]:', pos_pairs[i])            
            # raw_input('PAUSED')
            # print('--------------------------')

        for i in range(len(neg_pairs)):    # neg_pairs[i] = [0=subj_name1, 1=global_idx1, 2=subj_name2, 3=global_idx2]
            file_name_sample1 = all_pc_paths[neg_pairs[i][1]].split('/')[-2]
            file_name_sample2 = all_pc_paths[neg_pairs[i][3]].split('/')[-2]
            neg_pairs[i][1] = file_name_sample1
            neg_pairs[i][3] = file_name_sample2
            # print('neg_pairs[i]:', neg_pairs[i])            
            # raw_input('PAUSED')
            # print('--------------------------')

        return pos_pairs, neg_pairs


    def make_pairs_labels_with_paths(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples=True):
        pos_pairs, neg_pairs = self.make_pairs_global_indexes(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples, num_pos_pairs, num_neg_pairs, reuse_samples)
        pos_pair_label = '1'
        neg_pair_label = '0'

        all_pos_pairs_paths = []
        all_neg_pairs_paths = []

        for i, pos_pair in enumerate(pos_pairs):
            # print('pair '+str(i)+'/'+str(len(pos_pairs)), '   pos_pair:', pos_pair)
            subj_name, index1, index2 = pos_pair
            path_sample1 = all_pc_paths[index1]
            path_sample2 = all_pc_paths[index2]
            all_pos_pairs_paths.append((pos_pair_label, path_sample1, path_sample2))
            # print('all_pos_pairs_paths[-1]:', all_pos_pairs_paths[-1])            
            # raw_input('PAUSED')
            # print('--------------------------')

        for i, neg_pair in enumerate(neg_pairs):
            # print('pair '+str(i)+'/'+str(len(pos_pairs)), '   neg_pair:', neg_pair)
            subj_name1, index1, subj_name2, index2 = neg_pair
            path_sample1 = all_pc_paths[index1]
            path_sample2 = all_pc_paths[index2]
            all_neg_pairs_paths.append((neg_pair_label, path_sample1, path_sample2))
            # print('all_neg_pairs_paths[-1]:', all_neg_pairs_paths[-1])            
            # raw_input('PAUSED')
            # print('--------------------------')

        # return all_pos_pairs_paths, all_neg_pairs_paths
        return all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label


    def save_pairs_txt_file_lfw_format(self, pos_pairs_format_lfw, neg_pairs_format_lfw, perc_train, perc_test, pairs_file_path):
        with open(pairs_file_path, 'w') as file:
            num_pos_pairs_train = int(round(len(pos_pairs_format_lfw) * perc_train))
            num_neg_pairs_train = int(round(len(neg_pairs_format_lfw) * perc_train))
            file.write(str(num_pos_pairs_train) + '\ttrain positive pairs' + '\n')
            for i, pos_pair in enumerate(pos_pairs_format_lfw[:num_pos_pairs_train]):
                file.write(str(pos_pair[0]) + '\t' + str(pos_pair[1]) + '\t' + str(pos_pair[2]) + '\n')

            file.write(str(num_neg_pairs_train) + '\ttrain negative pairs' + '\n')
            for i, neg_pair in enumerate(neg_pairs_format_lfw[:num_neg_pairs_train]):
                file.write(str(neg_pair[0]) + '\t' + str(neg_pair[1]) + '\t' + str(neg_pair[2]) + '\t' + str(neg_pair[3]) + '\n')

            num_pos_pairs_test = int(round(len(pos_pairs_format_lfw) * perc_test))
            num_neg_pairs_test = int(round(len(neg_pairs_format_lfw) * perc_test))
            file.write(str(num_pos_pairs_test) + '\ttest positive pairs' + '\n')
            for i, pos_pair in enumerate(pos_pairs_format_lfw[num_pos_pairs_test:]):
                file.write(str(pos_pair[0]) + '\t' + str(pos_pair[1]) + '\t' + str(pos_pair[2]) + '\n')

            file.write(str(num_neg_pairs_test) + '\ttest negative pairs' + '\n')
            for i, neg_pair in enumerate(neg_pairs_format_lfw[num_neg_pairs_test:]):
                file.write(str(neg_pair[0]) + '\t' + str(neg_pair[1]) + '\t' + str(neg_pair[2]) + '\t' + str(neg_pair[3]) + '\n')


    def check_duplicate_pairs(self, pos_pairs, neg_pairs):
        num_repeated_pos_pairs = 0
        num_repeated_neg_pairs = 0
        for pair_search in pos_pairs:
            found_indexes = [idx for idx, pair in enumerate(pos_pairs) if pair[1] in pair_search and pair[2] in pair_search]
            num_repeated_pos_pairs += len(found_indexes) - 1  # consider only repetitions
            if len(found_indexes) > 1:
                duplicate_pairs = [pos_pairs[fi] for fi in found_indexes]
                print('pair_search:', pair_search)
                print('duplicate_pairs:', duplicate_pairs)
                print('found_indexes:', found_indexes)
                print('----------------')

        for pair_search in neg_pairs:
            found_indexes = [idx for idx, pair in enumerate(neg_pairs) if pair[1] in pair_search and pair[3] in pair_search]
            num_repeated_pos_pairs += len(found_indexes) - 1  # consider only repetitions
            if len(found_indexes) > 1:
                duplicate_pairs = [neg_pairs[fi] for fi in found_indexes]
                print('pair_search:', pair_search)
                print('duplicate_pairs:', duplicate_pairs)
                print('found_indexes:', found_indexes)
                print('----------------')

        print('pos_pairs:', len(pos_pairs), '    num_repeated_pos_pairs:', num_repeated_pos_pairs)
        print('neg_pairs:', len(neg_pairs), '    num_repeated_neg_pairs:', num_repeated_neg_pairs)


def save_list_to_file(lines, file_path):
    try:
        with open(file_path, 'w') as file:
            file.writelines("%s\n" % line for line in lines)
    except IOError:
        print("An error occurred while writing to the file.")


def read_file_lines(file_path):
    lines = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
    except FileNotFoundError:
        print("File not found.")
    except IOError:
        print("An error occurred while reading the file.")    
    return lines


def clear_strings_list(strs_list, substring_to_remove):
    for i in range(len(strs_list)):
        # print('strs_list[i]:', strs_list[i])
        strs_list[i] = strs_list[i].replace(substring_to_remove, '')
        # print('substring_to_remove:', substring_to_remove)
        # print('strs_list[i]:', strs_list[i])
        # print('-------------')
    return strs_list


def split_list_get_indices(list=[], num_parts=10):
    start = 0
    end = 0
    part_size = int(round(len(indexes_samples) / num_parts_compress))
    # print('part_size:', part_size)
    subindexes_samples = []
    for i in range(num_parts_compress-1):
        end = start + part_size
        # print('start:', start, '    end:', end)
        subindexes_samples.append([start, end])
        start = end
    
    end = len(indexes_samples)
    # print('start:', start, '    end:', end)
    subindexes_samples.append([start, end])
    return subindexes_samples


def convert_list_of_strings_to_list_of_lists(lst):
    cleaned_list = [string.strip('[] ') for string in lst]
    elements = [string.split(',') for string in cleaned_list]
    integer_list = [[int(element) for element in sublist] for sublist in elements]
    return integer_list


def make_tarfile(output_filename, paths_files):
    with tarfile.open(output_filename, "w:gz") as tar:
        for i, path_file in enumerate(paths_files):
            print(f'{i}/{len(paths_files)-1} - compressing file:', path_file)
            print('output_filename:', output_filename)
            tar.add(path_file)
            print('-------------------------')


def compress_all_files_in_parts(output_dir, base_output_file_name, paths_files, indexes_samples, parts_indexes):
    num_total_compressed_files = 0
    for i, part_idx in enumerate(parts_indexes):
        indexes_subj_paths = indexes_samples[part_idx[0]:part_idx[1]]
        # print(f'part {i+1}/{len(parts_indexes)} - part_idx:', part_idx)
        # print(f'    indexes_subj_paths:', indexes_subj_paths)
        start_path_idx_part = indexes_subj_paths[0][0]
        end_path_idx_part = indexes_subj_paths[-1][-1]+1
        paths_files_part = paths_files[start_path_idx_part:end_path_idx_part]
        num_total_compressed_files += len(paths_files_part)
        # print(f'    start_path_idx_part:', start_path_idx_part, '    end_path_idx_part:', end_path_idx_part)
        # print('    paths_files_part:', paths_files_part)
        # print()

        compressed_part_file_path = output_dir + '/' + base_output_file_name + f'_part{i}' + '.tar.gz'
        print(f'Compressing part {i} ...')
        make_tarfile(compressed_part_file_path, paths_files_part)
        print()
    print('num_total_compressed_files:', num_total_compressed_files)



if __name__ == '__main__':
    base_dir = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1'

    dataset_folder = 'images'
    # dataset_folder = 'images_22subj'
    # dataset_folder = 'images_1000subj'

    dataset_path = base_dir + '/' + dataset_folder

    dir_level=2
    file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'

    output_dir = '/datasets1/bjgbiesseck/MS-Celeb-1M/ms1m-retinaface-t1/3D_reconstruction/MS-Celeb-1M_3D_reconstruction_originalMICA'

    # num_parts_compress = 1
    # num_parts_compress = 2
    # num_parts_compress = 3
    # num_parts_compress = 5
    num_parts_compress = 10


    ########### PROCESSING ############
    os.chdir(base_dir)
    pointcloud_paths_file = base_dir + '/' + dataset_folder + '_pointcloud_paths'  + '.txt'
    indexes_samples_file = base_dir + '/' + dataset_folder + '_indexes_samples'  + '.txt'

    if not os.path.isfile(pointcloud_paths_file) and not os.path.isfile(indexes_samples_file):
        print('Searching all files ending with \'' + file_ext + '\'...')
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples = TreeMS1MV3_3DReconstructedMICA().get_all_pointclouds_paths_count(dataset_path, dir_level=dir_level, pc_ext=file_ext)
        # print('all_pc_paths:', all_pc_paths)
        # print('all_pc_subjects:', all_pc_subjects)
        # for i, pc_path in enumerate(all_pc_paths):
        #     print(f'{i}/{len(all_pc_paths)-1} - pc_path: {pc_path}')

        print('Saving pointcloud paths in:', pointcloud_paths_file)
        save_list_to_file(all_pc_paths, pointcloud_paths_file)
        print('Done!\n')

        print('Saving indexes samples in:', indexes_samples_file)
        save_list_to_file(indexes_samples, indexes_samples_file)
        print('Done!\n')
    
    else:
        print('Loading pointcloud paths from:', pointcloud_paths_file)
        all_pc_paths = read_file_lines(pointcloud_paths_file)
        # for i, pc_path in enumerate(all_pc_paths):
        #     print(f'{i}/{len(all_pc_paths)-1} - pc_path: {pc_path}')
        print('Done!\n')

        print('Loading indices paths from:', indexes_samples_file)
        indexes_samples = read_file_lines(indexes_samples_file)
        indexes_samples = convert_list_of_strings_to_list_of_lists(indexes_samples)
        # for i, pc_path in enumerate(all_pc_paths):
        #     print(f'{i}/{len(all_pc_paths)-1} - pc_path: {pc_path}')
        print('Done!\n')

    print('Cleaning paths...')
    all_pc_paths = clear_strings_list(all_pc_paths, base_dir + '/')
    # for i, pc_path in enumerate(all_pc_paths):
    #     print(f'{i}/{len(all_pc_paths)-1} - pc_path: {pc_path}')
    print('Done!\n')
    # print('indexes_samples:', indexes_samples)

    print('Spliting paths...')
    parts_indexes = split_list_get_indices(indexes_samples, num_parts_compress)
    # print('parts_indexes:', parts_indexes)
    print('Done!\n')
    # for idx in parts_indexes:
    #     print('indexes_samples[idx[0]:idx[1]]:', indexes_samples[idx[0]:idx[1]])

    print('Compressing pointclouds...')
    compress_all_files_in_parts(output_dir, dataset_folder, all_pc_paths, indexes_samples, parts_indexes)
    print('Done!\n')
