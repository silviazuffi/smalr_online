"""

"""


from os.path import exists, join, dirname, basename, splitext
from os import makedirs
from glob import glob
import numpy as np
import chumpy as ch
import cv2
import pickle as pkl

from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb

from mycore.io import load_animal_model, load_keypoints, get_matlab_anno, load_seg
from mycore.camera import setup_camera
from util.myrenderer import render_mesh
from estimate_global_pose import estimate_global_pose
from joint_limits_prior import LimitPrior
from mycore.io import get_anno_path, load_keymapping
from pose_prior import Prior
from joint_limits_prior import LimitPrior
from animal_shape_prior import MultiShapePrior

def get_landmarks(anno_path, keymapping_name='tiger_wtail_wears_new_tailstart'):
    import scipy.io as sio
    res = sio.loadmat(anno_path, squeeze_me=True, struct_as_record=False)
    res = res['annotation']
    kp = res.kp.astype(float)

    invisible = res.invisible
    vis = np.atleast_2d(~invisible.astype(bool)).T
    landmarks = np.hstack((kp, vis))
    names = [str(res.names[i]) for i in range(len(res.names))]

    return landmarks, names

def load_results(pkl_path, model):
    from smpl_webuser.verts import verts_decorated as SmplModel
    with open(pkl_path, 'rb') as f:
        res = pkl.load(f, encoding='latin1')

    sv = SmplModel(
            trans=ch.array(res['trans']),
            pose=ch.array(res['pose']),
            v_template=model.v_template,
            J=model.J,
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style = 'lbs',
            f=model.f,
            betas=ch.array(res['betas']),
            shapedirs=model.shapedirs[:,:,:len(res['betas'])])

    if 'E' in res.keys():
        E = res['E']
    else:
        E = None
    if 'params' in res.keys():
        params = res['params']
    else:
        params = None
    return sv, res['kp'][0], res['kp'][1], res['flength'], E, res['landmarks'], res['cam_t'], res['cam_rt'], res['cam_k'], params


def get_annotation(img_path, anno_path, model, cam, img, viz=True, offset=0, keymapping_name='tiger_wtail_wears_new_tailstart', im_scale=1.0):

    landmarks, names = get_landmarks(anno_path, keymapping_name)
    landmarks[:,:2] = landmarks[:,:2]*im_scale
   
    landmarks[:,:2] += offset
    keypoints = landmarks[:,:2]
    keypoint_vids, part_names = load_keymapping(keymapping_name)

    rot, trans = estimate_global_pose(landmarks, keypoint_vids, model, cam, img, viz=viz)
    fix_rot = True
    model.pose[:3] = rot
    model.trans[:] = trans

    return model, keypoints, keypoint_vids, fix_rot, landmarks, names

