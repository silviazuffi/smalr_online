import cv2
import pickle as pkl
from os.path import join, exists, splitext, basename
from os import makedirs
from scipy.signal import medfilt
import chumpy as ch
from util.myrenderer import render_mesh, render_orth, trim_sides
from util.image import resize_img
from mycore.camera import setup_camera
from track_frame_fit import load_results
import numpy as np

from psbody.mesh.meshviewer import MeshViewer, MeshViewers
from psbody.mesh import Mesh

def body_load(body, model, code):

    for ind in range(len(body)):
        save_name = body[ind]['save_name']
        if code is not None:
            save_name = save_name.replace('.pkl', '_' + code + '.pkl')
        sv, kp_proj, key_vids, flength, E, landmarks, cam_t, cam_rt, cam_k = load_results(save_name, model)
        body[ind]['isGT'] = True
        body[ind]['E'] = E
        body[ind]['landmarks'] = landmarks
        body[ind]['key_vids'] = key_vids
        body[ind]['pose'] = sv.pose.r.copy()
        body[ind]['betas'] = sv.betas.r.copy()
        body[ind]['trans'] = sv.trans.r.copy()
        body[ind]['flength'] = flength.copy()
        body[ind]['cam_t'] = cam_t
        body[ind]['cam_rt'] = cam_rt
        body[ind]['cam_k'] = cam_k

    return body


def body_save(body, body_imgs, model, unused, code, Ind=None, sv_result=None, save_variables=True):
    '''
    '''
    from smpl_webuser.verts import verts_decorated as SmplModel

    if Ind is None:
        Ind = range(len(body))
    else:
        Ind = [Ind]
    for ind in Ind:
        img = body_imgs[ind]['img']
        pose = body[ind]['pose']
        betas = body[ind]['betas']
        trans = body[ind]['trans']
        flength = body[ind]['flength']
        key_vids = body[ind]['key_vids']
        landmarks = body[ind]['landmarks']
        cam_f = body[ind]['flength']
        cam_t = body[ind]['cam_t'] 
        cam_rt = body[ind]['cam_rt']
        cam_k = body[ind]['cam_k']
        img_offset = body_imgs[ind]['img_offset'] 
        img_scale = body_imgs[ind]['img_scale'] 
        img_path = body_imgs[ind]['img_path'] 
        cam_c = np.zeros(2)
        cam_c[0] = np.array(img.shape[1])/2.0
        cam_c[1] = np.array(img.shape[0])/2.0
        E = body[ind]['E']
        nbetas = len(betas)

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


        cam = setup_camera(img.shape[0], img.shape[1])
        cam.k[:] = cam_k
        cam.f[:] = cam_f
        cam.t[:] = cam_t
        cam.rt[:] = cam_rt
        cam.c[:] = cam_c

        cam.v = sv 

        save_name = body[ind]['save_name']

        if code is None:
            save_img_path = save_name.replace('.pkl', '.png')
            save_path = save_name
        else:
            save_path = save_name.replace('.pkl', '_' + code + '.pkl')
            save_img_path = save_name.replace('.pkl', '_' + code + '.png')
        
        if save_variables:    
            with open(save_path, 'wb') as f:
               pkl.dump({
                   'pose': pose, 'betas': betas, 'flength': cam_f,
                   'cam_rt':cam_rt, 'cam_t':cam_t, 'cam_k':cam_k,
                   'trans': trans, 'kp': (key_vids, key_vids),
                   'landmarks': landmarks, 'E':E}, f)
        mesh = Mesh(sv.r, model.f)
        img0 = 255*np.ones_like(img)
        img_res = render_mesh(mesh, img.shape[1], img.shape[0], cam, img=img,  world_frame=True)
        img_result = np.hstack((img, img_res * 255.))

        cv2.imwrite(save_img_path, img_result)

    return
