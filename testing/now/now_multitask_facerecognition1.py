# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os, sys
import time
from glob import glob
from shutil import copyfile

# logs = '/home/wzielonka/projects/MICA/testing/now/logs/'   # original
# jobs = '/home/wzielonka/projects/MICA/testing/now/jobs/'   # original
# root = '/home/wzielonka/projects/MICA/output/'             # original
logs = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/logs/'     # Bernardo
jobs = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/jobs/'     # Bernardo
root = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/'               # Bernardo

# experiments = []   # original
# experiments = ['4_mica_duo_TESTS_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100']   # Bernardo
# experiments = ['5_mica_duo_TESTS_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedCOSFACE=glint360k-r100']   # Bernardo
# experiments = ['6_mica_duo_TESTS_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100']   # Bernardo
# experiments = ['10_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5']   # Bernardo
# experiments = ['11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0']   # Bernardo
experiments = ['27_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_maskface=True_lamb1=1.0_lamb2=1.0']   # Bernardo


def test():
    global experiments

    # BERNARDO
    if not os.path.exists(root):
        print('BERNARDO: creating dir \'' + root + '\' ... ', end='')
        os.mkdir(root)
        print(' done!')
    if not os.path.exists(logs):
        print('BERNARDO: creating dir \'' + logs + '\' ... ', end='')
        os.mkdir(logs)
        print(' done!')
    if not os.path.exists(jobs):
        print('BERNARDO: creating dir \'' + jobs + '\' ... ', end='')
        os.mkdir(jobs)
        print(' done!')

    if len(experiments) == 0:
        experiments = list(filter(lambda f: 'condor' not in f, os.listdir('../../output/')))

    # os.system('rm -rf logs')
    # os.system('rm -rf jobs')

    os.makedirs('logs', exist_ok=True)
    os.makedirs('jobs', exist_ok=True)

    for experiment in sorted(experiments):
        print(f'Testing {experiment}')
        copyfile(f'{root}{experiment}/model.tar', f'{root}{experiment}/best_models/best_model_last.tar')
        for idx, checkpoint in enumerate(glob(root + experiment + f'/best_models/*.tar')):
            model_name = checkpoint.split('/')[-1].split('.')[0]              # Original
            # model_name = '.'.join(checkpoint.split('/')[-1].split('.')[0:-1])   # Bernardo

            # TEST
            print('model_name:', model_name)
            print('checkpoint:', checkpoint)
            # sys.exit(0)

            model_name = model_name.replace('best_model_', 'now_test_')
            predicted_meshes = f'{root}{experiment}/{model_name}/predicted_meshes/'
            run = f'{experiment}_{str(idx).zfill(5)}'

            # Bernardo
            print('model_name:', model_name)
            print('predicted_meshes:', predicted_meshes)
            print('run:', run)

            '''
            # original
            with open(f'{jobs}/{run}.sub', 'w') as fid:
                fid.write('executable = /bin/bash\n')

                # arguments = f'/home/wzielonka/projects/MICA/testing/now/template.sh {experiment} {checkpoint} now {predicted_meshes}'    # original
                arguments = f'/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/template_multitask_facerecognition1.sh {experiment} {checkpoint} now {predicted_meshes}'      # BERNARDO

                fid.write(f'arguments = {arguments}\n')
                fid.write(f'error = {logs}{run}.err\n')
                fid.write(f'output = {logs}{run}.out\n')
                fid.write(f'log = {logs}{run}.log\n')
                fid.write(f'request_cpus = 4\n')
                fid.write(f'request_gpus = 1\n')
                fid.write(f'requirements = (TARGET.CUDAGlobalMemoryMb > 5000) && (TARGET.CUDAGlobalMemoryMb < 33000)\n')
                fid.write(f'request_memory = 8192\n')
                fid.write(f'queue\n')
            '''

            # Bernardo
            # arguments = f'/home/wzielonka/projects/MICA/testing/now/template.sh {experiment} {checkpoint} now {predicted_meshes}'    # original
            # arguments = f'/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/template.sh {experiment} {checkpoint} now {predicted_meshes}'      # BERNARDO
            arguments = f'/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/testing/now/template_multitask_facerecognition1.sh {experiment} {checkpoint} now {predicted_meshes}'      # BERNARDO

            # os.system(f'condor_submit_bid 512 {jobs}/{run}.sub')   # original
            # os.system(f'condor_submit {jobs}/{run}.sub')             # Bernardo
            os.system(f'/bin/bash {arguments}')

            time.sleep(2)


if __name__ == '__main__':
    test()
