''' Util for matching using silhouettes '''
''' Author: Angjoo Kawazawa '''
''' Modified by Silvia Zuffi '''

import numpy as np
import chumpy as ch
from opendr.renderer import ColoredRenderer
import cv2
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from util.myrenderer import render_mesh
from opendr.camera import ProjectPoints
from sbody.robustifiers import GMOf

def scalecam(cam, imres):
    # Returns a camera which shares camera parameters by reference,
    # and whose scale is resized according to imres
    from chumpy import array
    from opendr.camera import ProjectPoints
    return ProjectPoints(
        rt=cam.rt,
        t=cam.t,
        f=cam.f * imres,
        c=array(cam.c.r * imres),
        k=cam.k)


def fit_silhouettes_pyramid_multi_model(objs,
                            sv,
                            silhs,
                            cams,
                            #weights=1.,
                            mv=None,
                            fix_shape=False,
                            imgs=None,
                            fix_rot=False,
                            fix_trans=False,
                            s2m_weights=1.,
                            m2s_weights=1.,
                            max_iter=100,
                            alpha=None, vc=None, symIdx=None, mv2=None, FIX_POSE=False):
    ''' if alpha is not None, then replace that with sv.betas'''
    from opendr.camera import ProjectPoints

    silhs = [np.uint8(silh[:, :, 0] > 0) for silh in silhs]

    # Setup silhouet term camera.
    cam_copy = [ProjectPoints(
        rt=cam.rt, t=cam.t, f=cam.f, c=cam.c, k=cam.k, v=cam.v) for cam in cams]

    if imgs[0].shape[1] < 900:
        scales = 1. / (2 * np.array([3, 2, 1, 0.5]))
    else:
        scales = 1. / (2 * np.array([6, 4, 3, 2, 1]))

    res_silh = []

    for sc in scales:

        if 'shape_prior' in objs.keys():
            objs['shape_prior'] = 0.4 * objs['shape_prior'] 

        silh_here = [cv2.resize(silh, (int(silh.shape[1] * sc),int(silh.shape[0] * sc))) for silh in silhs]
        cam_here = [scalecam(cam, sc) for cam in cam_copy]
        for i,cam in enumerate(cam_copy):
            cam_here[i].v = cam.v
        print('Scale %g' % (1 / sc))
        w_s2m = s2m_weights
        w_m2s = m2s_weights
        R, s_objs = fit_silhouettes_multi_model(objs, sv, silh_here, cam_here,
                                        w_s2m, w_m2s, max_iter, mv, fix_shape,
                                        cams, imgs, alpha=alpha, fix_trans=fix_trans,
                                        pyr_scale=sc, vc=vc, symIdx=symIdx, mv2=mv2, FIX_POSE=FIX_POSE)

        # For scales < 1 we optimize f on the kp_camera (cams) and then we update cam_copy
        for i in range(len(cams)):
            cam_copy[i].f[:] = cam_here[i].f.r/sc 
        res_silh.append(R)

    # Compute energy
    E = 0
    for term in s_objs.values():
        E = E + np.mean(term.r)

    return res_silh, E

def fit_silhouettes_multi_model(objs,
                    sv,
                    silhs,
                    cameras,
                    w_s2m=10,
                    w_m2s=20,
                    max_iter=100,
                    mv=None,
                    fix_shape=False,
                    kp_camera=None,
                    imgs=None, alpha=None,
                    fix_trans=False, pyr_scale=1.0, vc=None, symIdx=None, mv2=None, FIX_POSE=False):

    nCameras = len(cameras)
    # Projected sihlouette
    #dist = [np.mean(sv[i].r, axis=0)[2]  for i in range(nCameras)]

    frustums = [{'near': ch.min(sv[i], axis=0)[2],
           'far': ch.max(sv[i], axis=0)[2],
           'width': silhs[i].shape[1],
           'height': silhs[i].shape[0]} for i,camera in enumerate(cameras)]

    rends = [ColoredRenderer(
        vc=np.ones_like(sv[i].r),
        v=sv[i],
        f=sv[i].f,
        camera=cameras[i],
        frustum=frustums[i],
        bgcolor=ch.array([0, 0, 0])) for i in range(nCameras)]

    # silhouette error term (model-to-scan)
    #obj_m2s, obj_s2m, dist_tsf = setup_silhouette_obj(silhs, rends, sv[0].model.f)
    obj_m2s, obj_s2m, dist_tsf = setup_silhouette_obj(silhs, rends, sv[0].f)

    if mv is not None:
        import matplotlib.pyplot as plt
        plt.ion()

        def on_step(_):
            for i in range(len(rends)):
                rend = rends[i]
                img = imgs[i]
                edges = cv2.Canny(np.uint8(rend[:, :, 0].r * 255), 100, 200)
                coords = np.array(np.where(edges > 0)).T
                plt.figure(200+i, facecolor='w')
                plt.clf()
                plt.subplot(2, 2, 1)
                plt.imshow(dist_tsf[i])
                plt.plot(coords[:, 1], coords[:, 0], 'w.')
                plt.axis('off')
                plt.subplot(2, 2, 2)
                plt.imshow(rend[:, :, 0].r)
                plt.title('fitted silhouette')
                plt.draw()
                plt.axis('off')
                if img is not None and kp_camera is not None:
                    plt.subplot(2, 2, 3)
                    plt.imshow(img[:, :, ::-1])
                    plt.scatter(kp_camera[i].r[:, 0], kp_camera[i].r[:, 1])
                    plt.axis('off')
                plt.subplot(2, 2, 4)
                plt.imshow((silhs[i]+rend[:, :, 0].r)/2.0)
                plt.axis('off')
                plt.draw()
                plt.show(block=False)
                plt.pause(1e-5)
                if vc is not None:
                    vc1 = vc[i].r.copy()
                    vc1[:,0] = vc[i].r.copy()[:,2]
                    vc1[:,2] = vc[i].r.copy()[:,0]
                    mv[i].set_dynamic_meshes([Mesh(sv[i].r, sv[i].model.f, vc=vc1)])
                    vc2 = vc[i].r.copy()[symIdx,:]
                    vc2[:,0] = vc[i].r.copy()[symIdx,2]
                    vc2[:,2] = vc[i].r.copy()[symIdx,0]
                    mv2[i].set_dynamic_meshes([Mesh(sv[i].r, sv[i].model.f, vc=vc2)])
                else:
                    mv[i].set_dynamic_meshes([Mesh(sv[i].r, sv[i].f)])

    else:
        on_step = None

    new_objs = objs.copy()
    for i in range(nCameras):
        new_objs['s2m_'+str(i)] = obj_s2m(w_s2m, i)
        new_objs['m2s_'+str(i)] = obj_m2s(w_m2s, i)

    print('weights: s2m %.2f m2s %.2f' % (w_s2m, w_m2s))

    nbetas = len(sv[0].betas.r)

    free_variables = []
    if fix_trans is False:
        for i in range(nCameras):
            free_variables.append(sv[i].trans) 
    else:
        free_variables = []
    if fix_shape is False:
        if alpha is not None:
            free_variables.append(alpha)
        else:
            if not fix_shape:
                for i in range(nCameras):
                    free_variables.append(sv[i].betas)

    for i in range(nCameras):
        if not FIX_POSE:
            free_variables.append(sv[i].pose)

    # If objective contains 'feq', then add cam.f to free variables.
    if 'feq' in new_objs:
        for i in range(len(rends)):   
            free_variables.append(cameras[i].f)

    opt = {'maxiter': max_iter, 'e_3': 1e-2}

    if max_iter > 0:
        ch.minimize(new_objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)

    def render_and_show(sv):
        for i in range(len(rends)):
            img = imgs[i]
            img_res = render_mesh(Mesh(sv[i].r, sv[i].model.f), img.shape[1], img.shape[0], kp_camera[i], near=0.5, far=20)
            plt.figure()
            plt.imshow(img[:, :, ::-1])
            plt.imshow(img_res)
            plt.axis('off')    

    return rends[0].r, new_objs

def setup_silhouette_obj(silhs, rends, f):
    n_model = [ch.sum(rend[:, :, 0] > 0) for rend in rends]
    dist_tsf = [cv2.distanceTransform(np.uint8(1 - silh), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for silh in silhs]

    # Make sigma proportional to image area.
    # This gives radius 400 for 960 x 1920 image.
    sigma_ratio = 0.2727
    sigma = [np.sqrt(sigma_ratio * (silh.shape[0] * silh.shape[1]) / np.pi) for silh in silhs]
    # Model2silhouette ==> Consistency (contraction)

    def obj_m2s(w, i):
        return w * (GMOf(rends[i][:, :, 0] * dist_tsf[i], sigma[i]) / np.sqrt(n_model[i]))

    # silhouette error term (scan-to-model) ==> Coverage (expansion)
    coords = [np.fliplr(np.array(np.where(silh > 0)).T) + 0.5  for silh in silhs]# is this offset needed?
    scan_flat_v = [np.hstack((coord, ch.zeros(len(coord)).reshape((-1, 1)))) for coord in coords]
    scan_flat = [Mesh(v=sflat_v, f=[]) for sflat_v in scan_flat_v]
    # 2d + all 0.
    sv_flat = [ch.hstack((rend.camera, ch.zeros(len(rend.v)).reshape((-1, 1)))) for rend in rends]
    for i in range(len(rends)):
        sv_flat[i].f = f

    def obj_s2m(w, i):
        from sbody.mesh_distance import ScanToMesh
        return w * ch.sqrt(GMOf(ScanToMesh(scan_flat[i], sv_flat[i], f), sigma[i]))
       
    # For vis
    for i in range(len(rends)):
        scan_flat[i].vc = np.tile(np.array([0, 0, 1]), (len(scan_flat[i].v), 1))

    return obj_m2s, obj_s2m, dist_tsf
