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

import sys                  # Bernardo
from abc import ABC
from glob import glob
from pathlib import Path

import numpy as np
from pytorch3d.io import load_objs_as_meshes

# from datasets.creation.instances.instance import Instance   # original
from instances.instance import Instance                       # Bernardo


class FRGC(Instance, ABC):
    def __init__(self):
        super(FRGC, self).__init__()
        # self.dst = '/scratch/NFC/OnFlame/FRGC/'   # original
        # self.src = '/scratch/NFC/FRGC_v2/'        # original
        self.dst = '/MICA/OnFlame/FRGC/'            # Bernardo
        self.src = '/MICA/FRGC/'                    # Bernardo
        # self.img_ext = '.ppm'

    def get_images(self):
        # BERNARDO
        print('\nFRGC: get_images()')

        images = {}
        for actor in sorted(glob(self.get_src() + 'images/*')):
            imgs = list(filter(lambda f: 'Spring2003range' not in f, glob(f'/{actor}/*/*.jpg')))    # original
            # imgs = list(filter(lambda f: 'Spring2003range' not in f, glob(f'*{self.img_ext}')))   # Bernardo

            # TESTE
            print('FRGC(): get_images(): imgs:', imgs)
            # tqdm.write('FRGC(): get_images(): imgs:', imgs)
            # sys.exit(0)

            images[Path(actor).name] = imgs

        return images

    def get_flame_params(self):
        prams = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_parameters/*')):
            prams[Path(actor).name] = glob(f'/{actor}/*.npz')

        return prams

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'registrations/*')):
            registrations[Path(actor).name] = glob(f'/{actor}/*.obj')

        return registrations

    def get_meshes(self):
        meshes = {}
        for file in sorted(glob(self.get_src() + 'registrations_tmp_new/*')):
            meshes[Path(file).name] = glob(f'/{file}/*.obj')

        sessions = np.load('/home/wzielonka/documents/scans_to_session.npy', allow_pickle=True)[()]
        valid = []
        for key in sessions.keys():
            if 'Spring2003range' not in sessions[key]:
                valid.append(key)

        filtered = {}
        for actor in meshes.keys():
            files = meshes[actor]
            selected = list(filter(lambda f: Path(f).stem in valid, files))
            if len(selected) > 0:
                filtered[actor] = selected

        return filtered

    def transform_mesh(self, path):
        self.update_obj(path[0])
        mesh = load_objs_as_meshes(path, device=self.device)
        mesh.scale_verts_(10.0)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]

        return mesh.clone()
