import glob
import os
import sys


def check_corresponding_files(original_file_paths, original_ext_search, target_path, target_ext):
    missing_corresponding = []
    for i, file_path in enumerate(original_file_paths):
        original_file_name = file_path.split('/')[-1]
        target_file_name = original_file_name.replace(original_ext_search, target_ext)
        target_file_path = target_path + '/' + target_file_name
        if not os.path.isfile(target_file_path):
            missing_corresponding.append(target_file_name)
    return missing_corresponding


def load_file_paths(path, ext_search):
    file_paths = sorted(glob.glob(path + '/*' + ext_search))
    # print(len(file_paths))
    # print(file_paths[:10])
    return file_paths


def main(original_path, original_ext_search, facedetect_path, facedetect_ext_search):
    original_file_paths = load_file_paths(original_path, original_ext_search)
    missing_corresponding_files = check_corresponding_files(original_file_paths, original_ext_search, facedetect_path, facedetect_ext_search)

    if len(missing_corresponding_files) > 0:
        print('Missing files in \'' + facedetect_path + '\':')
        for i, miss_file in enumerate(missing_corresponding_files):
            print(i, ':', miss_file)
        print('Total:', len(missing_corresponding_files))
    else:
        print('No missing files!')

if __name__ == '__main__':
    original_imgs_path = '/home/bjgbiesseck/datasets/CelebA/Img/img_align_celeba'
    facedetect_imgs_path = '/home/bjgbiesseck/GitHub/MICA/demo/arcface/CelebA/Img/img_align_celeba'

    original_ext_search = '.jpg'
    target_ext_search = '.jpg'

    main(original_imgs_path, original_ext_search, facedetect_imgs_path, target_ext_search)