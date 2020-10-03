from os.path import exists, join, dirname, basename, splitext
import cv2
import numpy as np
import chumpy as ch
from chumpy import Ch
from pose_prior import Prior
from joint_limits_prior import LimitPrior
from animal_shape_prior import MultiShapePrior
from sbody.robustifiers import GMOf
from multiclip_model_fit import set_pose_prior, set_pose_prior_tail, set_limit_prior, set_shape_prior #, set_params
from smalr_settings import settings


def set_params(DO_FREE_SHAPE=False, SOLVE_FLATER=True, FIX_SHAPE=False, nCameras=1, FREE_SHAPE_TYPE='allpca'):
    params = {}
    params['k_robust_sig'] = settings['k_robust_sig']
    params['k_kp_term'] = settings['ref_k_kp_term']
    return params


def set_pose_objs(sv, cam, landmarks, key_vids, animal=None, shape_data_name=None, nbetas=0, kp_weights=None, fix_rot=False, SOLVE_FLATER=True, FIX_CAM=False, ONLY_KEYP=False, OPT_SHAPE=True, DO_FREE_SHAPE=False):

    nCameras = len(cam)
    nClips = nCameras

    params = set_params(SOLVE_FLATER, nCameras)

    pose_prior = [set_pose_prior(len(sv[0].pose.r)) for _ in range(nClips)]
    pose_prior_tail = [set_pose_prior_tail(len(sv[0].pose.r))  for _ in range(nClips)]
    if OPT_SHAPE:
        shape_prior = [set_shape_prior(DO_FREE_SHAPE, animal, shape_data_name) for _ in range(nClips)]

    # indices with no prior
    noprior_ind = ~pose_prior[0].use_ind
    noprior_ind[:3] = False

    limit_prior = [set_limit_prior(len(sv[0].pose.r)) for _ in range(nClips)]

    init_rot = [sv[ic].pose[:3].r.copy() for ic in range(nClips)]
    init_trans = [sv[ic].trans.r.copy() for ic in range(nClips)]
    init_pose = [sv[ic].pose.r.copy() for ic in range(nClips)]

    # Setup keypoint projection error with multi verts:
    j2d = [None]*nCameras
    assignments = [None]*nCameras
    num_points = [None]*nCameras
    use_ids = [None]*nCameras
    visible_vids = [None]*nCameras
    all_vids = [None]*nCameras

    for i in range(nCameras):
        visible = landmarks[i][:, 2].astype(bool)

        use_ids[i] = [id for id in np.arange(landmarks[i].shape[0]) if visible[id]]
        visible_vids[i] = np.hstack([key_vids[i][id] for id in use_ids[i]])

        group = np.hstack([index * np.ones(len(key_vids[i][row_id])) for index, row_id in enumerate(use_ids[i])])
        assignments[i] = np.vstack([group == j for j in np.arange(group[-1]+1)])
        num_points[i] = len(use_ids[i])

        all_vids[i] = visible_vids[i]
        cam[i].v = sv[i][all_vids[i], :]
        j2d[i] = landmarks[i][use_ids[i], :2]

        if kp_weights is None:
            kp_weights = np.ones((landmarks[i].shape[0], 1))

    def kp_proj_error(i, w, sigma):
        return w * kp_weights[use_ids[i]] * GMOf(ch.vstack([cam[i][choice] if np.sum(choice) == 1 else cam[i][choice].mean(axis=0) for choice in assignments[i]]) - j2d[i], sigma) / np.sqrt(num_points[i])

    objs = {}
    for i in range(nCameras):
        objs['kp_proj_'+str(i)] = kp_proj_error(i, params['k_kp_term'], params['k_robust_sig'])
        if not ONLY_KEYP:
            objs['trans_init_'+str(i)] = params['k_trans_term'] * (sv[i].trans - init_trans[i])

        if not ONLY_KEYP:
            if fix_rot:
                objs['fix_rot_'+str(i)] = params['k_rot_term'] * (sv[i].pose[:3] - init_rot[i])
        if OPT_SHAPE:
            if (i > 0):
                objs['betas_var_'+str(i)] = params['betas_var']*ch.abs(sv[i-1].betas-sv[i].betas)
            objs['shape_prior_'+str(i)] = shape_prior[i](sv[i].betas) / np.sqrt(nbetas)

    if not FIX_CAM:
        for i in range(nCameras):
            objs['feq_'+str(i)] = 1e3 * (cam[i].f[0] - cam[i].f[1])
            objs['fpos_'+str(i)] = 1e3 * ch.maximum(0, 500-cam[i].f[0])
        if not SOLVE_FLATER:
            for i in range(nCameras):
                objs['freg_'+str(i)] = 9 * 1e2 * (cam[i].f[0] - 3000) / 1000.
                objs['cam_t_pos_'+str(i)] =  1e3 * ch.maximum(0, 0.01-sv[i].trans[2])
                del objs['trans_init_'+str(i)]

    num_pose_prior = len(pose_prior[0](sv[0].pose))
    num_limit_prior = len(limit_prior[0](sv[0].pose))

    if not ONLY_KEYP:
        if np.sum(noprior_ind) > 0:
            objs['rest_poseprior_'+str(i)] = params['k_rest_pose_term'] * (sv[i].pose[noprior_ind] - init_pose[noprior_ind]) / np.sqrt(len(sv[i].pose[noprior_ind]))
        for i in range(nClips):
            objs['pose_limit_'+str(i)] = params['k_limit_term'] * limit_prior[i](sv[i].pose) / np.sqrt(num_limit_prior)
            objs['pose_prior_'+str(i)] = pose_prior[i](sv[i].pose) / np.sqrt(num_pose_prior)
            objs['pose_prior_tail_'+str(i)] = 2.0 *pose_prior_tail[i](sv[i].pose) / np.sqrt(num_pose_prior)

    return objs, params, j2d



