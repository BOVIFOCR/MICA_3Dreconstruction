from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path


# BERNARDO
class LFW_Verif_Pairs_Images:
    
    def load_pairs_samples_protocol_from_file(self, protocol_file_path='pairs.txt', dataset_path='lfw', file_ext='.jpg'):
        pos_pair_label = '1'
        neg_pair_label = '0'
        
        with open(protocol_file_path, 'r') as fp:
            all_lines = [line.strip('\n') for line in fp.readlines()]
            # print('all_lines:', all_lines)

            line = all_lines[0].strip('\n').split('\t')
            num_folds, num_pair_type_per_fold = int(line[0]), int(line[1])
            num_total_pairs = num_folds * num_pair_type_per_fold * 2

            all_pairs = [ None ] * num_total_pairs
            # for i in range(num_total_pairs):
            l = 1    # Ignores first line
            for f in range(num_folds):

                # positive pairs
                for p in range(num_pair_type_per_fold):
                    line = all_lines[l]
                    # print(f'{l}/{num_total_pairs} line: {line}')
                    subj0, id0, id1 = line.split('\t')
                    path_sample0 = dataset_path + '/' + subj0 + '/' + subj0 + '_' + id0.zfill(4) + file_ext
                    path_sample1 = dataset_path + '/' + subj0 + '/' + subj0 + '_' + id1.zfill(4) + file_ext
                    pair_label = pos_pair_label
                    all_pairs[l-1] = (path_sample0, path_sample1, pair_label)
                    l += 1
                
                # negative pairs
                for p in range(num_pair_type_per_fold):
                    line = all_lines[l]
                    subj0, id0, subj1, id1 = line.split('\t')
                    # print(f'{l}/{num_total_pairs} line: {line}')
                    path_sample0 = dataset_path + '/' + subj0 + '/' + subj0 + '_' + id0.zfill(4) + file_ext
                    path_sample1 = dataset_path + '/' + subj1 + '/' + subj1 + '_' + id1.zfill(4) + file_ext
                    pair_label = neg_pair_label
                    all_pairs[l-1] = (path_sample0, path_sample1, pair_label)
                    l += 1

            # for i in range(len(all_pairs)):
            #     print(f'lfw_verif_pairs_imgs - load_pairs_samples_protocol_from_file - all_pairs[{i}]: {all_pairs[i]}')

            return all_pairs, pos_pair_label, neg_pair_label


    


    def make_cropped_lfw_dataset(self, dataset_path='lfw', input_file_ext='.jpg', output_path='lfw_cropped_aligned', output_file_ext='.png'):
        import cv2
        from skimage.io import imread, imsave
        from skimage.transform import estimate_transform, warp
        from tqdm import tqdm
        from insightface.app import FaceAnalysis
        from insightface.app.common import Face
        from insightface.utils import face_align
        
        def detect_crop_face(img, app):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
            i = 0
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            aimg = face_align.norm_crop(img, landmark=face.kps)
            aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
            return aimg

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.4)
        actors = sorted(os.listdir(dataset_path))
        for i, actor in enumerate(actors):
            output_path_folder = output_path + '/' + actor
            if os.path.isdir(dataset_path+'/'+actor) and not os.path.exists(output_path_folder):
                os.makedirs(output_path_folder)
            image_paths = sorted(glob(dataset_path + '/' + actor + '/*' + input_file_ext))
            for image_path in image_paths:
                print('------------------------')
                print(f'lfw_verif_pairs_imgs - make_cropped_lfw_dataset - {i}/{len(actors)-1} - {actor} - image_path: {image_path}')
                image = imread(image_path)[:, :, :3]
                print(f'lfw_verif_pairs_imgs - make_cropped_lfw_dataset - {i}/{len(actors)-1} - {actor} - image.shape: {image.shape}')
                print(f'lfw_verif_pairs_imgs - make_cropped_lfw_dataset - detecting face...')
                croped_face_img = detect_crop_face(image, app)
                output_image_path = output_path_folder + '/' + image_path.split('/')[-1].replace(input_file_ext, output_file_ext)
                print(f'lfw_verif_pairs_imgs - make_cropped_lfw_dataset - {i}/{len(actors)-1} - {actor} - croped_face_img.shape: {croped_face_img.shape}')
                imsave(output_image_path, croped_face_img)
                print(f'lfw_verif_pairs_imgs - make_cropped_lfw_dataset - {i}/{len(actors)-1} - {actor} - output_image_path: {output_image_path}')

                # input('PAUSED')
                # sys.exit(0)




if __name__ == '__main__':
    # protocol_file_path='/datasets1/bjgbiesseck/lfw/pairs.txt'
    # dataset_path = '/datasets1/bjgbiesseck/lfw'
    # file_ext='.jpg'
    # all_pairs, pos_pair_label, neg_pair_label = LFW_Verif_Pairs_Images().load_pairs_samples_protocol_from_file(protocol_file_path, dataset_path, file_ext)
    # # for i in range(len(all_pairs)):
    # #     print(f'lfw_verif_pairs_imgs - load_pairs_samples_protocol_from_file - all_pairs[{i}]: {all_pairs[i]}')

    dataset_path = '/datasets1/bjgbiesseck/lfw'
    output_path = '/datasets1/bjgbiesseck/lfw_cropped_aligned'
    LFW_Verif_Pairs_Images().make_cropped_lfw_dataset(dataset_path=dataset_path, input_file_ext='.jpg', output_path=output_path, output_file_ext='.png')
