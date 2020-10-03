''' Util for matching using silhouettes '''
''' Author: Angjoo Kawazawa '''
''' Modified by Silvia Zuffi '''

import numpy as np
import chumpy as ch
from opendr.renderer import ColoredRenderer
import cv2
import cv2 as cv
from psbody.mesh.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from util.myrenderer import render_mesh
from opendr.camera import ProjectPoints
from silhouette_multi_model import scalecam
from sbody.robustifiers import GMOf
from opendr.camera import ProjectPoints

def fit_silhouettes_pyramid_opt(objs,
                            shape_model, dv,
                            silhs,
                            cams,
                            j2d=None,
                            weights=1.,
                            mv=None,
                            imgs=None,
                            s2m_weights=1.,
                            m2s_weights=1.,
                            max_iter=100,
                            free_variables=[],
                            vc=None, symIdx=None, mv2=None, objs_pose=None):

    silhs = [np.uint8(silh[:, :, 0] > 0) for silh in silhs]

    # Setup silhouet term camera.
    cam_copy = [ProjectPoints(
        rt=cam.rt, t=cam.t, f=cam.f, c=cam.c, k=cam.k, v=cam.v) for cam in cams]

    if imgs[0].shape[1] < 900:
        scales = 1. / (2 * np.array([3, 2, 1, 0.5]))
        #scales = 1. / (2 * np.array([3, 2, 1]))
    else:
        scales = 1. / (2 * np.array([6, 4, 3, 2, 1]))

    res_silh = []

    for si, sc in enumerate(scales):

        silh_here = [cv2.resize(silh, (int(silh.shape[1] * sc),int(silh.shape[0] * sc))) for silh in silhs]
        cam_here = [scalecam(cam, sc) for cam in cam_copy]
        for i,cam in enumerate(cam_copy):
            cam_here[i].v = cam.v
        print('Scale %g' % (1 / sc))
        w_s2m = weights * s2m_weights
        w_m2s = weights * m2s_weights
        R, s_objs = fit_silhouettes_multi_model(objs, shape_model, dv, silh_here, cam_here,
                                        w_s2m, w_m2s, max_iter, free_variables, mv, 
                                        cams, imgs, j2d,
                                        pyr_scale=sc, 
                                        vc=vc, symIdx=symIdx, mv2=mv2, last_round=(si==len(scales)-1), objs_pose=objs_pose)

        # Silvia. For scales < 1 we optimize f on the kp_camera (cams) and then we update cam_copy
        for i in range(len(cams)):
            #cam_copy[i].f[:] = cams[i].f.r 
            cam_copy[i].f[:] = cam_here[i].f.r/sc 
        res_silh.append(R)

    # Compute energy
    E = 0
    for term in s_objs.values():
        E = E + np.mean(term.r)

    return dv.r #res_silh, E


def world_to_cam(Pw, camera):
    from opendr.geometry import Rodrigues
    R = Rodrigues(camera.rt)
    P = Pw.dot(R.T)+camera.t
    #P = (Pw - camera.t).dot(R)
    return P


def fit_silhouettes_multi_model(objs,
                    shape_model, dv,
                    silhs,
                    cameras,
                    w_s2m=10,
                    w_m2s=20,
                    max_iter=100,
                    input_free_variables=[],
                    mv=None,
                    kp_camera=None,
                    imgs=None, j2d=None,
                    pyr_scale=1.0, vc=None, symIdx=None, mv2=None, last_round=False, objs_pose=None):

    nCameras = len(cameras)

    k_annealing = 0.9
    #if last_round:
    #    k_annealing = 0.6
    if 'arap' in objs.keys():
        print('Downweighting regularizers ' + str(k_annealing))
        objs['arap'] = k_annealing*objs['arap']
    if 'sym_0' in objs.keys():
        objs['sym_0'] = k_annealing*objs['sym_0']
        objs['sym_1'] = k_annealing*objs['sym_1']
        objs['sym_2'] = k_annealing*objs['sym_2']
    if 'lap' in objs.keys():
        objs['lap'] = k_annealing*objs['lap']


    frustums = [{'near': ch.min(shape_model[i], axis=0)[2],
           'far': ch.max(shape_model[i], axis=0)[2],
           'width': silhs[i].shape[1],
           'height': silhs[i].shape[0]} for i,camera in enumerate(cameras)]


    rends = [ColoredRenderer(
        vc=np.ones_like(shape_model[i].r),
        v=shape_model[i],
        f=shape_model[i].f,
        camera=cameras[i],
        frustum=frustums[i],
        bgcolor=ch.array([0, 0, 0])) for i in range(nCameras)]

    '''
    import pylab
    pylab.figure()
    pylab.imshow(rends[0])
    pylab.show()
    import pdb; pdb.set_trace()
    pylab.figure()
    pylab.imshow(rends[1])
    pylab.show()
    import pdb; pdb.set_trace()
    #pylab.figure()
    #pylab.imshow(rends[2])
    #pylab.show()
    #import pdb; pdb.set_trace()
    '''

    # silhouette error term (model-to-scan)
    obj_m2s, obj_s2m, dist_tsf = setup_silhouette_obj(silhs, rends, shape_model[0].f)

    global c
    c = 0

    
    if True: #mv is not None:
        import matplotlib.pyplot as plt
        plt.ion()

        def on_step(_):
            global c
            mesh = [None]*len(rends)
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
                if img is not None and kp_camera is not None and j2d is not None:
                    plt.subplot(2, 2, 3)
                    plt.imshow(img[:, :, ::-1])
                    plt.scatter(kp_camera[i].r[:, 0], kp_camera[i].r[:, 1])
                    plt.scatter(j2d[i][:, 0], j2d[i][:, 1], c='w')
                    plt.axis('off')
                plt.subplot(2, 2, 4)
                plt.imshow((silhs[i]+rend[:, :, 0].r)/2.0)
                plt.axis('off')
                plt.draw()
                plt.show(block=False)
                plt.pause(1e-5)
                #mv[i].set_static_meshes([Mesh(shape_model[i].r, shape_model[i].f)])
                
                if False: #vc is not None:
                    vc1 = vc[i].r.copy()
                    vc1[:,0] = vc[i].r.copy()[:,2]
                    vc1[:,2] = vc[i].r.copy()[:,0]
                    mv[i].set_dynamic_meshes([Mesh(shape_model[i].r, shape_model[i].f, vc=vc1)])
                    vc2 = vc[i].r.copy()[symIdx,:]
                    vc2[:,0] = vc[i].r.copy()[symIdx,2]
                    vc2[:,2] = vc[i].r.copy()[symIdx,0]
                    mv2[i].set_dynamic_meshes([Mesh(shape_model[i].r, shape_model[i].f, vc=vc2)])
                else:
                    vc[:,0] = 0.8
                    vc[:,1] = 0.76
                    vc[:,2] = 0.77
                    mesh[i] = Mesh(shape_model[i].r, shape_model[i].f, vc=vc)
                    v = mesh[i].v.copy()
                    mesh[i].v[:,1] = -v[:,1]
                    mesh[i].v[:,2] = -v[:,2]

                    #mv[i].set_static_meshes([mesh])
                mv2[0][i].set_static_meshes([mesh[i]])
                #if c==0:
                #    import pdb; pdb.set_trace()

                #mv2[0][i].save_snapshot('tmp_%.4d.png' % (c))
            c = c+1
    else:
        on_step = None

    new_objs = objs.copy()
    for i in range(nCameras):
        new_objs['s2m_'+str(i)] = obj_s2m(w_s2m, i)
        new_objs['m2s_'+str(i)] = obj_m2s(w_m2s, i)
    if objs_pose is not None:
        new_objs_pose = objs_pose.copy()
        for i in range(nCameras):
            new_objs_pose['s2m_'+str(i)] = obj_s2m(w_s2m, i)
            new_objs_pose['m2s_'+str(i)] = obj_m2s(w_m2s, i)

    print('weights: s2m %.2f m2s %.2f' % (w_s2m, w_m2s))

    free_variables = []
    free_variables.append(dv)
    opt = {'maxiter': max_iter, 'e_3': 1e-2}
    if max_iter > 0:
        if len(input_free_variables) > 0:
            print('free_variables are pose')
            ch.minimize(new_objs_pose, x0=input_free_variables, method='dogleg', callback=on_step, options=opt)
        print('free_variables are dv')
        ch.minimize(new_objs, x0=free_variables, method='dogleg', callback=on_step, options=opt)

    def render_and_show(v):
        for i in range(len(rends)):
            img = imgs[i]
            img_res = render_mesh(Mesh(shape_model[i], f), img.shape[1], img.shape[0], cameras[i], near=0.5, far=20)
            plt.figure()
            plt.imshow(img[:, :, ::-1])
            plt.imshow(img_res)
            plt.axis('off')    

    return rends[0].r, new_objs


def setup_silhouette_obj(silhs, rends, f):
    n_model = [ch.sum(rend[:, :, 0] > 0) for rend in rends]

    #dist_tsf = [cv2.distanceTransform(np.uint8(1 - silh), cv.CV_DIST_L2, cv.CV_DIST_MASK_PRECISE) for silh in silhs]
    dist_tsf = [cv2.distanceTransform(np.uint8(1 - silh), cv.DIST_L2, cv.DIST_MASK_PRECISE) for silh in silhs]

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

