import os, sys
import numpy as np



# Bernardo
def find_best_treshold(pair_labels, cos_sims):
    best_tresh = 0
    best_acc = 0
    
    # start, end, step = 0, 1, 0.01
    start, end, step = 0, 4, 0.01    # used in insightface code

    treshs = np.arange(start, end+step, step)
    for i, tresh in enumerate(treshs):
        tresh = np.round(tresh, decimals=3)
        tp, fp, tn, fn, acc = 0, 0, 0, 0, 0
        for j, cos_sim in enumerate(cos_sims):
            pair_label = pair_labels[j]
            # print(j, '- pair_label:', pair_label, '   cos_sim:', cos_sim)
            if pair_label == 1:
                if cos_sim < tresh:
                    tp += 1
                else:
                    fn += 1
            else:
                if cos_sim >= tresh:
                    tn += 1
                else:
                    fp += 1

        acc = round((tp + tn) / (tp + tn + fp + fn), 4)
        # print(f'tester_multitask_FACEVERIFICATION - {i}/{treshs.size()[0]-1} - tresh: {tresh} - acc: {acc}')

        if acc > best_acc:
            best_acc = acc
            best_tresh = tresh

        print('\x1b[2K', end='')
        print(f'tester_multitask_FACEVERIFICATION - {i}/{len(treshs)-1} - tresh: {tresh}', end='\r')

    return best_tresh, best_acc



if __name__ == '__main__':

    # # LFW Dataset
    # file_model1 = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/19_mica_duo_pretrainedARCFACE=ms1mv3-r100_fr-feat=original-arcface_ORIGINAL-ARCFACE/cos-sims_checkpoint=_dataset=LFW.npy'
    # file_model2 = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/27_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_maskface=True_lamb1=1.0_lamb2=1.0/cos-sims_checkpoint=model_60000.tar_dataset=LFW.npy'

    # MLFW Dataset
    file_model1 = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/19_mica_duo_pretrainedARCFACE=ms1mv3-r100_fr-feat=original-arcface_ORIGINAL-ARCFACE/cos-sims_checkpoint=_dataset=MLFW.npy'
    file_model2 = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/27_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_maskface=True_lamb1=1.0_lamb2=1.0/cos-sims_checkpoint=model_60000.tar_dataset=MLFW.npy'

    # TALFW Dataset
    # file_model1 = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/19_mica_duo_pretrainedARCFACE=ms1mv3-r100_fr-feat=original-arcface_ORIGINAL-ARCFACE/cos-sims_checkpoint=_dataset=TALFW.npy'
    # file_model2 = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/27_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_maskface=True_lamb1=1.0_lamb2=1.0/cos-sims_checkpoint=model_60000.tar_dataset=TALFW.npy'

    # file_model1 = file_model2
    # file_model2 = file_model1


    data_model1 = np.load(file_model1, allow_pickle=True).item()
    data_model2 = np.load(file_model2, allow_pickle=True).item()

    sims_model1, pair_labels_model1 = data_model1['cos-sims'], data_model1['pair_labels']
    sims_model2, pair_labels_model2 = data_model2['cos-sims'], data_model2['pair_labels']
    
    # AVERAGE DISTANCE
    # final_sims = (sims_model1 + sims_model2) / 2

    # # MINIMUM DISTANCE
    # final_sims = np.zeros(shape=sims_model1.shape, dtype=float)
    # for i in range(len(final_sims)):
    #     final_sims[i] = np.minimum(sims_model1[i], sims_model2[i])

    # MAXIMUM DISTANCE
    final_sims = np.zeros(shape=sims_model1.shape, dtype=float)
    for i in range(len(final_sims)):
        final_sims[i] = np.maximum(sims_model1[i], sims_model2[i])

    print('\nFindind best treshold...')
    best_tresh, best_acc = find_best_treshold(pair_labels_model1, final_sims)
    print(f'\nbest_tresh: {best_tresh},   best_acc: {best_acc}')


