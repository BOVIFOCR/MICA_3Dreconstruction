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


import numpy as np
import torch

'''
# Original
from datasets.creation.generator import Generator
from datasets.creation.instances.bu3dfe import BU3DFE
from datasets.creation.instances.d3dfacs import D3DFACS
from datasets.creation.instances.facewarehouse import FaceWarehouse
from datasets.creation.instances.florence import Florence
from datasets.creation.instances.frgc import FRGC
from datasets.creation.instances.lyhm import LYHM
from datasets.creation.instances.pb4d import PB4D
from datasets.creation.instances.stirling import Stirling
'''

# BERNARDO
from generator import Generator
from instances.bu3dfe import BU3DFE
from instances.d3dfacs import D3DFACS
from instances.facewarehouse import FaceWarehouse
from instances.florence import Florence
from instances.frgc import FRGC
from instances.lyhm import LYHM
from instances.pb4d import PB4D
from instances.stirling import Stirling



np.random.seed(42)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # datasets = [FaceWarehouse(), LYHM(), D3DFACS(), FRGC(), Florence(), Stirling(), BU3DFE(), PB4D()]  # original
    # generator = Generator([FaceWarehouse()])                                                           # original

    # datasets = [FaceWarehouse(), LYHM(), FRGC(), Florence(), Stirling()]   # Bernardo
    # generator = Generator([FaceWarehouse()])                               # Bernardo

    datasets = [FaceWarehouse(), LYHM(), FRGC(), Florence(), Stirling()]   # Bernardo
    generator = Generator([FRGC()])                                        # Bernardo
    # generator = Generator([Stirling()])                                  # Bernardo
    # generator = Generator([FaceWarehouse()])                             # Bernardo
    # generator = Generator([LYHM()])                                      # Bernardo
    # generator = Generator([Florence()])                                  # NOT IMPLEMENTED YET

    # print('generator.run()')
    generator.run()
