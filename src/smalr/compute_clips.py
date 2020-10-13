import pdb

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import pickle as pkl
from os.path import join, exists, splitext, basename
from os import makedirs
from mycore.io import load_animal_model, load_keypoints, load_seg
from glob import glob
from multiclip_model_fit import multi_clip_model_fit_w_keypoints
from mycore.io import get_anno_path, load_keymapping
from mycore.camera import setup_camera
from track_frame_fit import get_annotation, load_results, get_annotation, get_landmarks
from util.myrenderer import render_mesh, render_orth, trim_sides
from util.image import resize_img
import chumpy as ch
from smalr_settings import settings
from body_utils import body_save, body_load

from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from psbody.mesh.meshviewer import MeshViewer, MeshViewers
from psbody.mesh import Mesh

def remove_tail(model):
    idx = pkl.load(open('tailIdx.pkl'))
    model.f = np.delete(model.f, idx, 0) 
    return model

def clean_seg_with_annotations(simg, anno_path, im_scale, keymapping_name):

    import scipy.ndimage as snd
    from scipy.stats import mode
    landmarks, names = get_landmarks(anno_path, keymapping_name)
    keypoints = landmarks[:,:2] * im_scale
    limg, n = snd.label(simg)

    # Which is the blob with the animal?
    idx = keypoints.astype('int')
    idx = idx[np.where(idx>0)[0][::2]]
    idx[:,[0,1]] = idx[:,[1,0]]
    labels = [limg[idx[j,0],idx[j,1]] for j in range(idx.shape[0])]
    L = np.asarray(labels)
    L = L[np.where(L>0)]
    lab = mode(L)[0]
    simg[np.where(limg!=lab)] = 0
    return simg


def compute_multiple_unsynch_frames_from_annotations(body, body_imgs, model, 
                                                     nbetas,
                                                     family, shape_data_name,
                                                     init_flength, frames,
                                                     viz=False, init_from_mean_pose=False,
                                                     FIX_SHAPE=False, DO_FREE_SHAPE=False,
                                                     symIdx=None,
                                                     opt_model_dir=None, orig_f=None,
                                                     COMPUTE_OPT=True,
                                                     FIX_CAM=False, FIX_TRANS=False, 
                                                     im_scale=None, keymapping_name=None):

    if init_from_mean_pose:
       mean_pose = np.load(settings['mean_pose_prior'])['mean_pose']

    nFrames = len(frames[0])
    nClips = len(body)
    cams = [None]*nClips

    for ind in range(nFrames):

        done = True
        for ic in range(nClips):
            if not exists(body[ic][ind]['save_name']):
                done = False
        
        if not done:
            save_results = True
            keypoints = [None]*nClips
            key_vids = [None]*nClips
            landmarks = [None]*nClips
            landmarks_names = [None]*nClips
            frame = [frames[ic][ind] for ic in range(nClips)]

            for ic in range(nClips):
                model.pose[:] = 0.0
                model.trans[:] = 0.0
                img = body_imgs[ic][ind]['img'] 
                seg = body_imgs[ic][ind]['seg'] 
                save_name = body[ic][ind]['save_name'] 
                anno_path = body[ic][ind]['anno_path'] 

                cams[ic] = setup_camera(img.shape[0], img.shape[1], flength=init_flength) 

                if init_from_mean_pose:
                    model.pose[3:] = mean_pose[3:]
                offset = body[ic][ind]['img_offset']

                model, keypoints[ic], key_vids[ic], _, landmarks[ic], landmarks_names[ic] = get_annotation(
                        frame[ic], anno_path, model, cams[ic], img, offset=offset, keymapping_name=keymapping_name, im_scale=im_scale[ic])

                FIX_POSE = False 

                body[ic][ind]['landmarks'] = landmarks[ic]
                body[ic][ind]['key_vids'] = key_vids[ic]
                body[ic][ind]['pose'] = model.pose.copy()
                body[ic][ind]['trans'] = model.trans.copy()
                body[ic][ind]['betas'] = model.betas.copy()
         
            imgs = [body_imgs[ic][ind]['img'] for ic in range(nClips)]
            segs = [body_imgs[ic][ind]['seg'] for ic in range(nClips)]
            body_ind = [body[ic][ind] for ic in range(nClips)]

            sv, img_final, params, final_f, E = multi_clip_model_fit_w_keypoints(body=body_ind,
                                                         model=model, cam=cams, landmarks=landmarks,
                                                         key_vids=key_vids, img=imgs, bbox=0,
                                                         nbetas=nbetas,
                                                         animal=family,
                                                         shape_data_name=shape_data_name,
                                                         seg=segs, viz=viz, FIX_SHAPE=FIX_SHAPE, DO_FREE_SHAPE=DO_FREE_SHAPE,
                                                         landmarks_names=landmarks_names, symIdx=symIdx,
                                                         FIX_CAM=FIX_CAM, FIX_TRANS=FIX_TRANS, FIX_POSE=FIX_POSE
                                                         )

            for ic in range(nClips):
                body[ic][ind]['isGT'] = True
                body[ic][ind]['E'] = E
                body[ic][ind]['pose'] = sv[ic].pose.r.copy()
                body[ic][ind]['betas'] = sv[ic].betas.r.copy()
                body[ic][ind]['trans'] = sv[ic].trans.r.copy()
                body[ic][ind]['flength'] = cams[ic].f.r.copy()
                body[ic][ind]['cam_t'] = cams[ic].t.r.copy()
                body[ic][ind]['cam_rt'] = cams[ic].rt.r.copy()
                body[ic][ind]['cam_k'] = cams[ic].k.r.copy()
                body[ic][ind]['params'] = params

            for ic in range(nClips): 
                body_save(body[ic], body_imgs[ic], model, cams[ic], code=None, Ind=ind,  sv_result=sv[ic])

        sv = [None]*nClips
        landmarks = [None]*nClips
        key_vids = [None]*nClips

        for ic in range(nClips): 
            save_name = body[ic][ind]['save_name']
            img = body_imgs[ic][ind]['img']
            sv[ic], kp_proj, key_vids[ic], flength, E, landmarks[ic], cam_t, cam_rt, cam_k, params = load_results(save_name, model)
            body[ic][ind]['landmarks'] = landmarks[ic]
            body[ic][ind]['key_vids'] = key_vids[ic]
            cams[ic] = setup_camera(img.shape[0], img.shape[1], flength=init_flength)
            cams[ic].k[:] = cam_k
            cams[ic].f[:] = flength
            cams[ic].t[:] = cam_t
            cams[ic].rt[:] = cam_rt

            body[ic][ind]['isGT'] = True
            body[ic][ind]['E'] = E
            body[ic][ind]['pose'] = sv[ic].pose.r.copy() 
            body[ic][ind]['betas'] = sv[ic].betas.r.copy()
            body[ic][ind]['trans'] = sv[ic].trans.r.copy() 
            body[ic][ind]['flength'] = cams[ic].f.r.copy() 
            body[ic][ind]['cam_t'] = cams[ic].t.r.copy()
            body[ic][ind]['cam_rt'] = cams[ic].rt.r.copy()
            body[ic][ind]['cam_k'] = cams[ic].k.r.copy()
            body[ic][ind]['params'] = params


        if opt_model_dir is not None:
            from optimize_smal import optimize_smal
            for ic in range(nClips):
                nbetas = len(body[ic][ind]['betas'])

            segs = [body_imgs[ic][ind]['seg'] for ic in range(nClips)]
            imgs = [body_imgs[ic][ind]['img'] for ic in range(nClips)]
            save_names = [body_imgs[ic][ind]['save_name'] for ic in range(nClips)]
            poses = [body[ic][ind]['pose'] for ic in range(nClips)]
            trans = [body[ic][ind]['trans'] for ic in range(nClips)]
            betas = [body[ic][ind]['betas'] for ic in range(nClips)]
            img_paths = [body_imgs[ic][ind]['img_path'] for ic in range(nClips)]
            img_offset = [body_imgs[ic][ind]['img_offset'] for ic in range(nClips)]
            img_scales = [body_imgs[ic][ind]['img_scale'] for ic in range(nClips)]

            # Compute texture on the whole original SMAL model topology
            model.f = orig_f
            _, names = get_landmarks(body[0][0]['anno_path'], keymapping_name)

            dv = optimize_smal(poses, trans, betas, model, cams, segs, imgs,
                               landmarks, names, key_vids, symIdx, ind, opt_model_dir,
                               save_names, COMPUTE_OPT=COMPUTE_OPT,
                               img_paths=img_paths, img_offset=img_offset, img_scales=img_scales)

    return body

def compute_clips(family, model_name, shape_data_name, base_dirs, save_base_dir,
                 animal_name, clips, frameStarts, frameStops,
                 symIdx, viz=False, init_flength=1000., init_from_mean_pose=False,
                 border=0, world_frame=False, opt_model_dir=None, custom_template=None,
                 NO_TAIL=False, code=None, landmarks=None,
                 max_image_size=-1):

    np.random.seed(0)
    if code is not None:
        method = code
    else:
        method = 'unknown'

    FIX_SHAPE = False
    COMPUTE_OPT = True
    if custom_template is not None:
        # Second round, we only compute the texture
        method = method + '_ct'
        opt_model_dir = opt_model_dir + '_ct'
        COMPUTE_OPT = False
        FIX_SHAPE = True

    nClips = len(clips)
    bodys = [None]*nClips
    bodys_imgs = [None]*nClips
    frames = [None]*nClips
    im_scale = [None]*nClips

    model = load_animal_model(model_name)

    orig_f = model.f.copy()

    if custom_template is not None:
        model.v_template[:] = custom_template

    if NO_TAIL:
        model = remove_tail(model)

    for ic in range(nClips):

        seq_name = animal_name[ic] + '_' + clips[ic]
        save_dir = join(save_base_dir, seq_name, 'tracking', method)

        if not exists(save_dir):
            makedirs(save_dir)

        img_dir = join(base_dirs[ic], seq_name)

        # Get all frames
        frames[ic] = sorted(glob(join(img_dir, '*.png')))

        # Select the range required
        start_id = frames[ic].index(frameStarts[ic])
        stop_id = frames[ic].index(frameStops[ic])
        frames[ic] = frames[ic][start_id:stop_id+1]

        # Initialize data struct that stores all. Bodies are initialized
        # from the GT annotations
        nFrames = len(frames[ic])
        print(nFrames)
        bodys[ic] = [None]*nFrames
        bodys_imgs[ic] = [None]*nFrames

        for ind, (frame) in enumerate(frames[ic]):
            save_name = join(save_dir, '%s.pkl' % splitext(basename(frame))[0])
            img = cv2.imread(frame)
            if np.max(img.shape) > max_image_size:
                im_scale[ic] = max_image_size / (1.0*np.max(img.shape))
            else:
                im_scale[ic] = 1.0
            print('Scaling images of: ' + str(im_scale[ic]))
            seg = load_seg(frame)
            if im_scale[ic] != 1.0:
                img = resize_img(img, im_scale[ic])
                seg = resize_img(seg, im_scale[ic])
            
            if True:
                if landmarks == 'face':
                    keymapping_name = 'tiger_wtail_wears_new_tailstart_wface'
                    anno_path = get_anno_path(frame, '_ferrari-tail-face.mat')
                elif landmarks == 'nose_htail':
                    keymapping_name = 'tiger_wtail_wears_new_tailstart_wnose_whtail'
                    anno_path = get_anno_path(frame, '_ferrari-tail-nose-htail.mat')
                else:
                    anno_path = get_anno_path(frame, '_ferrari-tail.mat')
                    keymapping_name = 'tiger_wtail_wears_new_tailstart'

                seg = clean_seg_with_annotations(seg, anno_path, im_scale[ic], keymapping_name)
                _, names = get_landmarks(anno_path, keymapping_name)
            else:
                anno_path = get_anno_path(frame, '.pkl')

            if border>0:
                w, h, d = seg.shape
                tmp = np.zeros((w+2*border, h+2*border, d), dtype=np.uint8)
                tmp[border:border+w, border:border+h, :] = seg
                seg = tmp.copy()
                w, h, d = img.shape
                tmp = np.zeros((w+2*border, h+2*border, d), dtype=np.uint8)
                tmp[border:border+w, border:border+h, :] = img
                img = tmp.copy()

            bodys_imgs[ic][ind] = {}
            bodys_imgs[ic][ind]['img'] = img
            bodys_imgs[ic][ind]['seg'] = seg
            bodys_imgs[ic][ind]['save_name'] = save_name 
            bodys_imgs[ic][ind]['img_path'] = frame
            bodys_imgs[ic][ind]['img_offset'] = border
            bodys_imgs[ic][ind]['img_scale'] = im_scale[ic]

            bodys[ic][ind] = {}
            bodys[ic][ind]['anno_path'] = anno_path
            bodys[ic][ind]['img_offset'] = border
            bodys[ic][ind]['save_name'] = save_name 

    if family == 'other':
        DO_FREE_SHAPE = True
        model.betas[:] = 0.
    else:
        # Load the mean shape of this family
        from animal_shape_prior import MultiShapePrior
        shape_prior = MultiShapePrior(family, shape_data_name)
        model.betas[:] = shape_prior.mu
        DO_FREE_SHAPE = False


    if custom_template is not None:
        model.betas[:] = 0.
    

    nbetas = 20

    bodys = compute_multiple_unsynch_frames_from_annotations(bodys, bodys_imgs, model,
        nbetas, family, shape_data_name, init_flength, frames, viz=viz, FIX_SHAPE=FIX_SHAPE, DO_FREE_SHAPE=DO_FREE_SHAPE,
        init_from_mean_pose=init_from_mean_pose, symIdx=symIdx,
        opt_model_dir=opt_model_dir, orig_f=orig_f, COMPUTE_OPT=COMPUTE_OPT,
        FIX_CAM=False, FIX_TRANS=False, im_scale=im_scale, keymapping_name=keymapping_name)


