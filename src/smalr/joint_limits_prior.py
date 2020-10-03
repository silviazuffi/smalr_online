import chumpy as ch
from chumpy import Ch
import pdb
import numpy as np
import cv2
import scipy.sparse as sps
import pickle as pkl
import time
import os

Ranges = {
          'pelvis': [[0,0],[0,0],[0,0]],
          'pelvis0': [[-0.3,0.3],[-1.2 ,0.5],[-0.1,0.1]],
          'spine': [[-0.4,0.4],[-1.0,0.9],[-0.8,0.8]],
          'spine0': [[-0.4,0.4],[-1.0,0.9],[-0.8,0.8]],
          'spine1': [[-0.4,0.4],[-0.5,1.2],[-0.4,0.4]],
          'spine3': [[-0.5,0.5],[-0.6,1.4],[-0.8,0.8]],
          'spine2': [[-0.5,0.5],[-0.4,1.4],[-0.5,0.5]],
          'RFootBack': [[-0.2,0.3],[-0.3,1.1],[-0.3,0.5]],
          'LFootBack': [[-0.3,0.2],[-0.3,1.1],[-0.5,0.3]],
          'LLegBack1': [[-0.2,0.3],[-0.5,0.8],[-0.5,0.4]],
          'RLegBack1': [[-0.3,0.2],[-0.5,0.8],[-0.4,0.5]],
          'Head': [[-0.5,0.5],[-1.0,0.9],[-0.9,0.9]], 
          'RLegBack2': [[-0.3,0.2],[-0.6,0.8],[-0.5,0.6]],
          'LLegBack2': [[-0.2,0.3],[-0.6,0.8],[-0.6,0.5]],
          'RLegBack3': [[-0.2,0.3],[-0.8,0.2],[-0.4,0.5]],
          'LLegBack3': [[-0.3,0.2],[-0.8,0.2],[-0.5,0.4]],
          'Mouth': [[-0.01,0.01],[-1.1,0],[-0.01,0.01]],
          'Neck': [[-0.8,0.8],[-1.0,1.0],[-1.1,1.1]],
          'LLeg1': [[-0.05,0.05],[-1.3,0.8],[-0.6,0.6]], # Extreme
          'RLeg1': [[-0.05,0.05],[-1.3,0.8],[-0.6,0.6]],
          'RLeg2': [[-0.05,0.05],[-1.0,0.9],[-0.6,0.6]], # Extreme
          'LLeg2': [[-0.05,0.05],[-1.0,1.1],[-0.6,0.6]], 
          'RLeg3': [[-0.1,0.4],[-0.3,1.4],[-0.4,0.7]], # Extreme
          'LLeg3': [[-0.4,0.1],[-0.3,1.4],[-0.7,0.4]],
          'LFoot': [[-0.3,0.1],[-0.4,1.5],[-0.7,0.3]], # Extreme
          'RFoot': [[-0.1,0.3],[-0.4,1.5],[-0.3,0.7]],
          'Tail7': [[-0.1,0.1],[-0.7,1.1],[-0.9,0.8]],
          'Tail6': [[-0.1,0.1],[-1.4,1.4],[-1.0,1.0]],
          'Tail5': [[-0.1,0.1],[-1.0,1.0],[-0.8,0.8]],
          'Tail4': [[-0.1,0.1],[-1.0,1.0],[-0.8,0.8]],
          'Tail3': [[-0.1,0.1],[-1.0,1.0],[-0.8,0.8]],
          'Tail2': [[-0.1,0.1],[-1.0,1.0],[-0.8,0.8]],
          'Tail1': [[-0.1,0.1],[-1.5,1.4],[-1.2,1.2]],
          'LEar': [[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]],
          'REar': [[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]],
}


class LimitPrior(object):
    def __init__(self, nPose):
        if nPose == 99:
            self.parts = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22}
        elif nPose == 105:
            self.parts = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'LEar':33, 'REar':34}
        else:
            import pdb; pdb.set_trace()
        self.id2name = {v: k for k, v in self.parts.items()}
        # Ignore the first joint.
        self.prefix = 3
        self.part_ids = np.array(sorted(self.parts.values()))
        # Margin so that the limit point is still low, without margin it will have 0.5.
        self.margin = 0.01
        self.min_values = np.hstack([np.array(np.array(Ranges[self.id2name[part_id]])[:, 0]) for part_id in self.part_ids])# - self.margin
        self.max_values = np.hstack([np.array(np.array(Ranges[self.id2name[part_id]])[:, 1]) for part_id in self.part_ids])# + self.margin
        self.ranges = Ranges

    def __call__(self, x):
        zeros = np.zeros_like(x[self.prefix:])
        res = ch.maximum(x[self.prefix: ] - self.max_values, zeros) + ch.maximum(self.min_values - x[self.prefix: ], zeros)
        return res

