from os.path import exists, join, dirname, basename, splitext
from os import makedirs
import cv2
import numpy as np
import chumpy as ch
from time import time
from smpl_webuser.serialization import load_model
from psbody.mesh.meshviewer import MeshViewer
from psbody.mesh import Mesh
from util.image import resize_img
from util.myrenderer import render_mesh, trim_sides, render_orth, render_tpose
from pose_prior import Prior
from joint_limits_prior import LimitPrior
from animal_shape_prior import MultiShapePrior
from smalr_settings import settings

from sbody.robustifiers import GMOf


def set_params(DO_FREE_SHAPE=False, SOLVE_FLATER=True, FIX_SHAPE=False, nCameras=1, img_size=-1, FREE_SHAPE_TYPE='allpca'):
    params = {}
    params['k_robust_sig'] = 150

    # Weights.
    params['k_kp_term'] = settings['k_kp_term']  
    params['k_shape_term'] = settings['k_shape_term']
    params['k_pose_term'] = settings['k_pose_term']
    params['k_tail_pose_term'] = settings['k_tail_pose_term']
    params['k_rest_pose_term'] = settings['k_rest_pose_term']
    params['k_limit_term'] = settings['k_limit_term']   
    if nCameras == 1:
        params['betas_var'] = 0
    else:
        params['betas_var'] = settings['k_betas_var']  

    if DO_FREE_SHAPE:
        params['k_pose_term'] = settings['k_pose_term_free_shape']
        params['k_shape_term'] =  settings['k_shape_term_free_shape'] 

    # Stay close to init term:
    params['k_trans_term'] = settings['k_trans_term']
    params['k_rot_term'] = settings['k_rot_term']

    # If we use silhouettes..
    params['k_m2s'] = settings['k_m2s']  
    params['k_s2m'] = settings['k_s2m']   

    params['opt_weights_only_kp'] = zip(params['k_pose_term'] * np.array([1, 8e-1, 5e-1, 4.9e-1]),
                  params['k_tail_pose_term'] * np.array([1, 8e-1, 5e-1, 4.9e-1]),
                  params['k_shape_term'] * np.array([1, .9, .7, .6]),)

    return params

def set_sv(DO_FREE_SHAPE, shape_data_name, model, nbetas, pose, trans, betas, FREE_SHAPE_TYPE='allpca'):

    from smpl_webuser.verts import verts_decorated as SmplModel
    sv = SmplModel(
            trans=ch.array(trans),
            pose=ch.array(pose),
            v_template=model.v_template,
            J=model.J,
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style = 'lbs',
            f=model.f,
            betas=ch.array(betas[:nbetas]),
            shapedirs=model.shapedirs[:,:,:nbetas])
    return sv

def set_pose_prior(poseDim):

    # Load pose prior:
    posep_dir = '../../pose_priors/'

    prior_path = join(posep_dir, settings['pose_prior'])
    pose_prior = Prior(prior_path)
    return pose_prior

def set_pose_prior_tail(poseDim):

    posep_dir = '../../pose_priors/'
    prior_path = join(posep_dir, settings['tail_pose_prior'])
    pose_prior = Prior(prior_path)
    return pose_prior


def set_limit_prior(poseDim):
    # Limit prior
    limit_prior = LimitPrior(poseDim)
    return limit_prior

def set_shape_prior(DO_FREE_SHAPE, animal=None, shape_data_name=None, FREE_SHAPE_TYPE='allpca'):
    # Multi animal shape prior
    if DO_FREE_SHAPE:
        def shape_prior(betas): return betas
    else:
        shape_prior = MultiShapePrior(animal, shape_data_name)
    return shape_prior

def multi_clip_model_fit_w_keypoints(body, model,
                     cam,
                     landmarks,
                     key_vids,
                     img,
                     bbox,
                     nbetas,
                     animal,
                     shape_data_name,
                     seg=None,
                     fix_rot=False,
                     viz=False,
                     DO_FREE_SHAPE=False,
                     SOLVE_FLATER=True,
                     FIX_SHAPE=False,
                     betas=None,
                     FIX_CAM=False,
                     maxiter=100,
                     params=None,
                     sv=None, pose_prior=None,
                     limit_prior=None,
                     shape_prior=None,
                     landmarks_names=None, init_flength=1000., 
                     symIdx=None, FREE_SHAPE_TYPE='allpca', FIX_TRANS=False, FIX_POSE=False):

    t0 = time()
    nCameras = len(cam)
    nClips = nCameras

    if params is None:
        params = set_params(DO_FREE_SHAPE, SOLVE_FLATER, FIX_SHAPE, nCameras, img[0].shape[1])

    if sv is None:
        sv = [set_sv(DO_FREE_SHAPE, shape_data_name, model, nbetas, body[ic]['pose'], body[ic]['trans'], body[ic]['betas']) for ic in range(nClips)]

    if betas is not None:
        for ic in range(nClips):
            sv.betas[ic][:] = betas

    if pose_prior is None:
        pose_prior = [set_pose_prior(len(model.pose.r)) for _ in range(nClips)]
        pose_prior_tail = [set_pose_prior_tail(len(model.pose.r))  for _ in range(nClips)]

    # indices with no prior
    noprior_ind = ~pose_prior[0].use_ind
    noprior_ind[:3] = False

    if limit_prior is None:
        limit_prior = [set_limit_prior(len(model.pose.r)) for _ in range(nClips)]

    if shape_prior is None:
        shape_prior = [set_shape_prior(DO_FREE_SHAPE, animal, shape_data_name) for _ in range(nClips)]

    init_rot = [body[ic]['pose'][:3] for ic in range(nClips)]
    init_trans = [body[ic]['trans'] for ic in range(nClips)]
    init_pose = [body[ic]['pose'] for ic in range(nClips)]

    # Setup keypoint projection error with multi verts:
    j2d = [None]*nCameras
    kp_weights = [None]*nCameras
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

        kp_weights[i] = np.ones((landmarks[i].shape[0], 1))
        kp_weights[i] = np.ones((landmarks[i].shape[0],1))

        
        kp_weights[i][landmarks_names[i].index('leftEye'),:] *= 2.
        kp_weights[i][landmarks_names[i].index('rightEye'),:] *= 2.
        kp_weights[i][landmarks_names[i].index('leftEar'),:] *= 2.
        kp_weights[i][landmarks_names[i].index('rightEar'),:] *= 2.
        if 'noseTip' in landmarks_names[i]:
            kp_weights[i][landmarks_names[i].index('noseTip'),:] *= 2.
        

    def kp_proj_error(i, w, sigma):
        return w * kp_weights[i][use_ids[i]] * GMOf(ch.vstack([cam[i][choice] if np.sum(choice) == 1 else cam[i][choice].mean(axis=0) for choice in assignments[i]]) - j2d[i], sigma) / np.sqrt(num_points[i])

    # Setup objective
    objs = {}

    for i in range(nCameras):
        objs['kp_proj_'+str(i)] = kp_proj_error(i, params['k_kp_term'], params['k_robust_sig'])
        objs['trans_init_'+str(i)] = params['k_trans_term'] * (sv[i].trans - init_trans[i])
        if fix_rot:
            objs['fix_rot_'+str(i)] = params['k_rot_term'] * (sv[i].pose[:3] - init_rot[i])

    free_variables = []
    if FIX_SHAPE:
        for ic in range(nClips):
            if not FIX_POSE:
                free_variables.append(sv[ic].pose)
            if not FIX_TRANS:
                free_variables.append(sv[ic].trans)
    else:
        for ic in range(nClips):
            if not FIX_POSE:
                free_variables.append(sv[ic].pose)
            if not FIX_TRANS:
                free_variables.append(sv[ic].trans)
            free_variables.append(sv[ic].betas)
            if ic > 0:
                objs['betas_var_'+str(ic)] = params['betas_var']*ch.abs(sv[ic-1].betas-sv[ic].betas)

    if not FIX_CAM:
        for i in range(nCameras):
            objs['feq_'+str(i)] = 1e3 * (cam[i].f[0] - cam[i].f[1])
            objs['fpos_'+str(i)] = 1e3 * ch.maximum(0, 500-cam[i].f[0])
    if not SOLVE_FLATER and not FIX_CAM:
        for i in range(nCameras):
            objs['freg_'+str(i)] = 9 * 1e2 * (cam[i].f[0] - 3000) / 1000.
            objs['cam_t_pos_'+str(i)] =  1e3 * ch.maximum(0, 0.01-sv[i].trans[2])
            del objs['trans_init_'+str(i)]
    else:
        for i in range(nCameras):
            objs['cam_t_pos_'+str(i)] =  1e3 * ch.maximum(0, 0.01-sv[i].trans[2])

    if not FIX_CAM:
        for i in range(nCameras):
            free_variables.append(cam[i].f)

    opt = {'maxiter': maxiter,'e_3': 1e-2}

    # Terms for normalization
    num_pose_prior = len(pose_prior[0](sv[0].pose))
    num_limit_prior = len(limit_prior[0](sv[0].pose))

    for i in range(nClips):
        print('flength initial: %f %f z:%f' %  (cam[i].f[0], cam[i].f[1], sv[i].trans[2]))

    if viz:
        import matplotlib.pyplot as plt
        mv = [MeshViewer(window_width=600, window_height=600) for i in range(nCameras)]
        mv2 = None
        for i in range(nCameras):
            mv[i].set_dynamic_meshes([Mesh(sv[i].r, model.f)])
            plt.ion()
            plt.figure(100+i, facecolor='w')
            plt.clf()

        def on_step(_):
            for i in range(nCameras):
                plt.figure(100+i, facecolor='w')
                plt.subplot(221)
                plt.cla()
                plt.imshow(seg[i][:, :, ::-1])
                plt.scatter(cam[i].r[:, 0], cam[i].r[:, 1])
                plt.scatter(j2d[i][:, 0], j2d[i][:, 1], c='w')
                plt.title('kp only')
                plt.axis('off')
                plt.draw()
                plt.pause(1e-1)
                plt.axis('off')
                mv[i].set_dynamic_meshes([Mesh(sv[i].r, model.f)])
    else:
        mv = None
        on_step = None

    if viz:
        on_step(None)

    pose_w_prev = 1.0
    shape_w_prev = 1.0
    for itr, (pose_w, tail_pose_w, shape_w) in enumerate(params['opt_weights_only_kp']):
        print('Iteration %d' % itr)
        print('shape_w ' + str(shape_w))

        for i in range(nClips):
            if not FIX_POSE:
                objs['pose_limit_'+str(i)] = params['k_limit_term'] * limit_prior[i](sv[i].pose) / np.sqrt(num_limit_prior)
                objs['pose_prior_'+str(i)] = pose_w * pose_prior[i](sv[i].pose) / np.sqrt(num_pose_prior)
                objs['pose_prior_tail_'+str(i)] = tail_pose_w * 2.0 *pose_prior_tail[i](sv[i].pose) / np.sqrt(num_pose_prior)
                if np.sum(noprior_ind) > 0:
                    objs['rest_poseprior_'+str(i)] = params['k_rest_pose_term'] * (sv[i].pose[noprior_ind] - init_pose[noprior_ind]) / np.sqrt(len(sv[i].pose[noprior_ind]))

        for i in range(nClips):
            objs['shape_prior_'+str(i)] = shape_w * shape_prior[i](sv[i].betas) / np.sqrt(nbetas)

        ch.minimize(objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)
        for i in range(nClips):
            print('beta here: %s'% ' '.join(['%.2f,' % v for v in sv[i].betas.r]))

    if viz:
        from util.image import plot_points
        for i in range(nCameras):
            plt.figure(100+i)
            img_kp = plot_points(cam[i].r, img[i])
            dist = np.mean(sv[i].r, axis=0)[2]
            img_res = render_mesh(Mesh(sv[i].r, model.f), img[i].shape[1], img[i].shape[0], cam[i], img=img[i], world_frame=True)
            plt.subplot(222)
            plt.imshow(img[i][:, :, ::-1])
            plt.imshow(img_res[:, :, ::-1])
            plt.axis('off')

    # Define a cost for matching vertex colors across different views
    vc = None
    E = 0
    if seg is not None:

        if viz:
            for i in range(nCameras):
                dist = np.mean(sv[i].r, axis=0)[2]
                img_res = render_mesh(Mesh(sv[i].r, model.f), img[i].shape[1], img[i].shape[0], cam[i], img=img[i], world_frame=True)
                plt.figure(100+i)
                plt.subplot(223)
                plt.imshow(img[i][:, :, ::-1])
                plt.axis('off')
                plt.scatter(j2d[i][:, 0], j2d[i][:, 1], c='w')
                plt.scatter(cam[i].r[:, 0], cam[i].r[:, 1])
                plt.title('+ silhouette')
                plt.subplot(224)
                plt.imshow(img[i][:, :, ::-1])
                plt.imshow(img_res[:, :, ::-1])

        # Get silhouette:
        from silhouette_multi_model import fit_silhouettes_pyramid_multi_model
        _, E = fit_silhouettes_pyramid_multi_model(objs, sv, seg, cam,
                                                   #params['k_silh_term'],
                                                   fix_shape=FIX_SHAPE,
                                                   mv=mv, imgs=img,
                                                   s2m_weights=params['k_s2m'],
                                                   m2s_weights=params['k_m2s'], fix_trans=FIX_TRANS,
                                                   vc=vc, symIdx=symIdx, mv2=mv2, FIX_POSE=FIX_POSE)
        if viz:
            for i in range(nCameras):
                dist = np.mean(sv[i].r, axis=0)[2]
                img_res = render_mesh(Mesh(sv[i].r, model.f), img[i].shape[1], img[i].shape[0], cam[i], img=img[i], world_frame=True)
                plt.figure(100+i)
                plt.subplot(223)
                plt.imshow(img[i][:, :, ::-1])
                plt.axis('off')
                plt.scatter(j2d[i][:, 0], j2d[i][:, 1], c='w')
                plt.scatter(cam[i].r[:, 0], cam[i].r[:, 1])
                plt.title('+ silhouette')
                plt.subplot(224)
                plt.imshow(img[i][:, :, ::-1])
                plt.imshow(img_res[:, :, ::-1])

    final_flength = cam[0].f.r.copy()
    for i in range(nCameras):
        print('flength: %.3f, %.3f' % (final_flength[0], final_flength[1]))

    if DO_FREE_SHAPE:
        use_color = 'others'
    else:
        use_color = animal

    margin = (80, 250)

    final_imgs = [None]*nCameras
    imgs_final = [None]*nCameras
    final_flengths = [None]*nCameras
    for i in range(nCameras):
        dist = np.mean(sv[i].r, axis=0)[2]
        im = img[i]
        img_final = render_mesh(Mesh(sv[i].r, model.f), im.shape[1], im.shape[0], cam[i], img=im, margin=margin, color_key=use_color, world_frame=True)
        imgs_final[i] = img_final
        img_rot0 = render_orth(sv[i], im.shape[1], im.shape[0], cam[i], deg=45, margin=margin, color_key=use_color)
        img_rot1 = render_orth(sv[i], im.shape[1], im.shape[0], cam[i], deg=-45, margin=margin, color_key=use_color)
        # Put together final image to save.
        img_pad = np.pad(im/255., ((margin[0], margin[0]),(0, 0),(0,0)), 'constant', constant_values=1.)
        img_pad_alph = np.ones((img_pad.shape[0],img_pad.shape[1],1)) 
        img_pad_alph[:margin[0], :, :] = 0
        img_pad_alph[img_pad.shape[0]-margin[0]:img_pad.shape[0], :, :] = 0

        img_final_alph = np.zeros((img_final.shape[0],img_final.shape[1])) 
        img_final_alph[margin[0]:-margin[0], margin[1]:-margin[1]] = 1
        img_final_alph = np.logical_or(img_final_alph, (~np.all(img_final == 1., axis=2)).astype(img_final.dtype))

        try:
            img_final0 = trim_sides(np.dstack((img_final, img_final_alph)))
        except:
            img_final0 = img_final
        img_pad = np.dstack((img_pad, img_pad_alph))
        try:
            final_img = np.hstack((img_pad, img_final0, trim_sides(img_rot1), trim_sides(img_rot0)))
        except:
            final_img = img_final0
        scale_factor = (500. / final_img.shape[0])
        final_img = resize_img(final_img, scale_factor)
        final_imgs[i] = final_img
        final_flengths[i] = cam[i].f.r

        if viz:
            plt.figure(300+i)
            plt.clf()
            plt.imshow(final_img)
            plt.axis('off')

    return sv, final_imgs, params, final_flengths, E
