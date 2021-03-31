import pickle as pkl
import numpy as np
from chumpy import Ch
import cv2
from psbody.mesh.colors import name_to_rgb

name2id33 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0}
#id2name33 = {v: k for k, v in name2id33.iteritems()}

name2id31 = {'RFoot': 12, 'RFootBack': 22, 'LLegBack1': 15, 'spine1': 2, 'Head': 14, 'RLegBack1': 19, 'RLegBack2': 20, 'RLegBack3': 21, 'LLegBack2': 16, 'LLegBack3': 17, 'spine3': 4, 'spine2': 3, 'Mouth': 30, 'Neck': 13, 'LFootBack': 18, 'LLeg1': 5, 'RLeg2': 10, 'RLeg3': 11, 'LLeg3': 7, 'RLeg1': 9, 'LLeg2': 6, 'spine': 1, 'LFoot': 8, 'Tail7': 29, 'Tail6': 28, 'Tail5': 27, 'Tail4': 26, 'Tail3': 25, 'Tail2': 24, 'Tail1': 23, 'root': 0}
#id2name31 = {v: k for k, v in name2id31.iteritems()}

name2id35 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0, 'LEar':33, 'REar':34}
#id2name35 = {v: k for k, v in name2id35.iteritems()}
id2name35 = {name2id35[key]: key for key in name2id35.keys()}

def get_ignore_names(path):
    if 'notail' in path:
        ignore_names = [key for key in name2id.keys() if 'Tail' in key]
    elif 'bodyneckelbowtail' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack']
    elif 'body_indept_limbstips' in path:
        ignore_names = ['Neck', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']

    elif 'prior_bodyelbow' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']

    elif 'bodyneckheadelbow' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    elif 'bodyneckelbow' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    elif 'bodyneck' in path:
        ignore_names = ['LLeg1', 'LLeg2', 'LLeg3', 'RLeg1', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack']
    elif 'backlegstail' in path:
        if '33parts' in path:
            ignore_names = ['root', 'RFoot', 'RFootBack', 'spine1', 'Head', 'pelvis0', 'spine0', 'spine3', 'spine2', 'Mouth', 'Neck', 'LFootBack', 'RLeg3', 'RLeg2', 'LLeg1', 'LLeg3', 'RLeg1', 'LLeg2', 'spine', 'LFoot']
        if '35parts' in path:
            ignore_names = ['root', 'RFoot', 'RFootBack', 'spine1', 'Head', 'pelvis0', 'spine0', 'spine3', 'spine2', 'Mouth', 'Neck', 'LFootBack', 'RLeg3', 'RLeg2', 'LLeg1', 'LLeg3', 'RLeg1', 'LLeg2', 'spine', 'LFoot', 'LEar', 'REar']
    elif '_body.pkl' in path:
        ignore_names = ['LLeg1', 'LLeg2', 'LLeg3', 'RLeg1', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'LLegBack1', 'LLegBack2', 'LLegBack3', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    else:
        ignore_names = []


    return ignore_names


class Prior(object):
    def __init__(self, prior_path):
        with open(prior_path, 'rb') as f:
            res = pkl.load(f, encoding='latin1') 
            
        self.precs = res['pic']
        self.mean = res['mean_pose']
        # Mouth closed!
        # self.mean[-2] = -0.4
        # Ignore the first 3 global rotation.
        prefix = 3
        if '33parts' in prior_path:
            pose_len = 99
            id2name = id2name33
            name2id = name2id33
        elif '35parts' in prior_path:
            pose_len = 105
            id2name = id2name35
            name2id = name2id35
        else:
            pose_len = 93
            id2name = id2name31
            name2id = name2id31

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False

        ignore_names = get_ignore_names(prior_path)

        if len(ignore_names) > 0:
            ignore_joints = sorted([name2id[name] for name in ignore_names])
            ignore_inds = np.hstack([np.array(ig * 3) + np.arange(0, 3)
                                     for ig in ignore_joints])
            self.use_ind[ignore_inds] = False

    def __call__(self, x):
        res = (x[self.use_ind] - self.mean).dot(self.precs)
        return res

