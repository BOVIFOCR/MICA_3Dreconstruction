import os, sys
import glob
import shutil
import numpy as np
from abc import ABCMeta, abstractmethod
import cv2


class DatasetOrganizer:
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):
        self.dataset_name = ''
        self.path_parent_dir = path_parent_dir
        self.subpath_input_dir = ''
        self.input_file_ext = ['']
        self.subpath_flame_parameters = ''
        self.subpath_registrations = ''
        self.subpath_output_images = ''
        self.output_dataset_name = output_dataset_name
        # print('DatasetOrganizer.__init__(): self.path_parent_dir:', self.path_parent_dir, '    self.output_dataset_name:', self.output_dataset_name)

    def load_image_paths_from_npy(self, npy_path=''):
        data = np.load(npy_path, allow_pickle=True).item()
        return data

    def concat_paths(self):
        self.subpath_npy_paths_mica = '../image_paths/' + self.dataset_name + '.npy'
        self.subpath_flame_parameters = self.output_dataset_name + '/' + self.dataset_name + '/FLAME_parameters'
        self.subpath_registrations = self.output_dataset_name + '/' + self.dataset_name + '/registrations'
        self.subpath_output_images = self.output_dataset_name + '/' + self.dataset_name + '/images'

        self.path_original_images  = self.path_parent_dir + '/' + self.dataset_name + '/' + self.subpath_input_dir
        self.path_flame_parameters = self.path_parent_dir + '/' + self.subpath_flame_parameters
        self.path_registrations    = self.path_parent_dir + '/' + self.subpath_registrations
        self.path_output_images    = self.path_parent_dir + '/' + self.subpath_output_images
        self.path_npy_paths_mica   = os.path.dirname(os.path.realpath(__file__)) + '/' + self.subpath_npy_paths_mica

    def print_folder_paths(self):
        print('self.path_original_images:', self.path_original_images)
        print('self.path_flame_parameters:', self.path_flame_parameters)
        print('self.path_registrations:', self.path_registrations)
        print('self.path_output_images:', self.path_output_images)
        print('self.path_npy_paths_mica:', self.path_npy_paths_mica)

    def create_subjects_folders(self, parent_dir='', subfolder_names=[]):
        for subfolder in subfolder_names:
            path_folder = parent_dir + '/' + subfolder
            # print('path_folder:', path_folder)
            os.makedirs(path_folder, exist_ok=True)

    def is_found_file_valid(self, subj, path_file):
        if '/'+subj in path_file or subj+'/' in path_file:
            return True
        return False

    def organize(self):
        paths_dict = self.load_image_paths_from_npy(self.path_npy_paths_mica)
        subj_names = list(paths_dict.keys())
        # print('paths_dict:', paths_dict)
        # print('subj_names:', subj_names)

        for i, subj in enumerate(subj_names):
            output_imgs = paths_dict[subj][0]    # ['M1044/M1044_001.jpg', 'M1044/M1044_002.jpg', etc]
            input_npz =   paths_dict[subj][1]    # 'M1044/m1044_NH.npz'
            # print('output_imgs:', output_imgs)
            # print('input_npz:', input_npz)
            # input('PAUSED')

            # COPY IMAGES
            for j, out_img in enumerate(output_imgs):
                file_name_to_search = out_img.split('/')[-1].split('.')[0]
                print('subj:', subj, '('+str(i+1)+'/'+str(len(subj_names))+')    out_img:', out_img, '    file_name_to_search:', file_name_to_search, '('+str(j+1)+'/'+str(len(output_imgs))+')')
                pattern_file_to_search = self.path_original_images + '/**/' + file_name_to_search + '.*'
                print('    pattern_file_to_search:', pattern_file_to_search)
                found_file = glob.glob(pattern_file_to_search, recursive=True)
                found_file = [f for f in found_file if self.is_found_file_valid(subj, f)]                  # Check if path contains subject name
                # found_file = [f for f in found_file if '/'+subj in f or subj+'/' in f]                  # Check if path contains subject name
                # found_file = [f for f in found_file if subj in f or '/'+subj in f or subj+'/' in f]     # Check if path contains subject name
                found_file = [f for ext in self.input_file_ext for f in found_file if f.endswith(ext)]    # Check extension of found files

                if len(found_file) == 0:
                    print('Error, file not found:', file_name_to_search)
                    sys.exit(0)
                elif len(found_file) > 1:
                    print('Error, multiple files found:', found_file)
                    sys.exit(0)
                found_file = found_file[0]
                print('    found_file:', '\''+found_file+'\'')
                # input('PAUSED')
                # sys.exit(0)

                if found_file.endswith(out_img.split('.')[-1]):
                    output_file = self.path_output_images + '/' + subj + '/' + out_img.split('/')[-1]
                else:
                    output_file = self.path_output_images + '/' + subj + '/' + file_name_to_search + '.jpg'

                assert output_file != found_file
                os.makedirs('/'.join(output_file.split('/')[:-1]), exist_ok=True)
                print('    copying output_file:', output_file)

                if found_file.endswith('.jpg') or found_file.endswith('.JPG') or \
                                                  found_file.endswith('.jpeg') or found_file.endswith('.JPEG') or \
                                                  found_file.endswith('.png') or found_file.endswith('.PNG'):
                    shutil.copyfile(found_file, output_file)
                else:
                    img_data = cv2.imread(found_file)
                    cv2.imwrite(output_file, img_data)

                if not os.path.exists(output_file):
                    print('Error, file not copied:', output_file)
                    sys.exit(0)

                # sys.exit(0)
                # input('PAUSED')
                print('-----------------')

            # sys.exit(0)



class DatasetOrganizer_FRGCv2(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):
        # self.path_parent_dir = path_parent_dir
        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'FRGC_v2'
        self.subpath_input_dir = 'FRGCv2.0/FRGC-2.0-dist/nd1'
        self.input_file_ext = ['.jpg', 'JPG', '.ppm']
        # self.subpath_output_images = 'MICA/FRGC/images'
        self.concat_paths()



class DatasetOrganizer_Stirling(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):
        # self.path_parent_dir = path_parent_dir
        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'STIRLING'
        self.subpath_input_dir = 'Stirling-ESRC_2D/Subset_2D_FG2018/HQ'
        self.input_file_ext = ['.jpg']
        # self.subpath_output_images = output_dataset_name + '/' + self.dataset_name + '/' + 'images'
        self.concat_paths()



class DatasetOrganizer_FaceWarehouse(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):
        # self.path_parent_dir = path_parent_dir
        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'FACEWAREHOUSE'
        self.subpath_input_dir = 'FaceWarehouse/FaceWarehouse_Data_0.part1'
        self.input_file_ext = ['.png']
        # self.subpath_output_images = output_dataset_name + '/' + self.dataset_name + '/' + 'images'
        self.concat_paths()



class DatasetOrganizer_LYHM(DatasetOrganizer):
    def __init__(self, path_parent_dir='/datasets1/bjgbiesseck', output_dataset_name='MICA'):
        # self.path_parent_dir = path_parent_dir
        super().__init__(path_parent_dir, output_dataset_name)
        self.dataset_name = 'LYHM'
        self.subpath_input_dir = 'headspacePngTka/subjects'
        self.input_file_ext = ['.png']
        # self.subpath_output_images = output_dataset_name + '/' + self.dataset_name + '/' + 'images'
        self.concat_paths()

    # overrides superclass method
    def is_found_file_valid(self, subj, path_file):
        subj_from_path = path_file.split('/')[-3]
        if ('/'+subj in path_file or subj+'/' in path_file) and subj == subj_from_path:
            return True
        return False



np.random.seed(42)

if __name__ == '__main__':

    path_parent_dir = '/datasets1/bjgbiesseck'
    output_dataset_name = 'MICA'

    # dataset_org = DatasetOrganizer_FRGCv2(path_parent_dir, output_dataset_name)
    # dataset_org = DatasetOrganizer_Stirling(path_parent_dir, output_dataset_name)
    # dataset_org = DatasetOrganizer_FaceWarehouse(path_parent_dir, output_dataset_name)
    dataset_org = DatasetOrganizer_LYHM(path_parent_dir, output_dataset_name)
    # dataset_org = DatasetOrganizer_Florence(path_parent_dir, output_dataset_name)        # TODO

    # dataset_org.print_folder_paths()
    dataset_org.organize()
