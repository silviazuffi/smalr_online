import cv2
import numpy as np
import chumpy as ch
from time import time
from util.myrenderer import render_mesh
from psbody.mesh import Mesh


def estimate_translation(landmarks, key_vids, focal_length, model):
    '''
    Estimates the z of the model using similar triangles.
    '''
    import itertools
    use_names = ['neck','leftShoulder', 'rightShoulder', 'backLeftKnee', 'backRightKnee', 'tailStart', 'frontLeftKnee', 'frontRightKnee']

    if model.pose[1] == np.pi/2:
        use_names = ['neck','frontLeftKnee', 'frontRightKnee']
        use_names += ['frontLeftAnkle', 'frontRightAnkle']

    # Redefining part names here..
    part_names = ['leftEye', 'rightEye', 'chin', 'frontLeftFoot', 'frontRightFoot', 'backLeftFoot', 'backRightFoot', 'tailStart', 'frontLeftKnee', 'frontRightKnee', 'backLeftKnee', 'backRightKnee', 'leftShoulder', 'rightShoulder', 'frontLeftAnkle', 'frontRightAnkle', 'backLeftAnkle', 'backRightAnkle', 'neck', 'TailTip']
    use_ids0 = [part_names.index(name) for name in use_names]
    # Only keep visible one
    visible = landmarks[:, 2].astype(bool)
    keypoints = landmarks[:, :2]
    use_ids = [id for id in use_ids0 if visible[id]]
    if len(use_ids) < 2:
        print('ERROR: Not enough points visible..')
        import ipdb; ipdb.set_trace()
        use_names = ['neck','frontLeftKnee', 'frontRightKnee']
        use_names += ['frontLeftAnkle', 'frontRightAnkle']
        use_ids0 = [part_names.index(name) for name in use_names]
        use_ids = [id for id in use_ids0 if visible[id]]

    pairs = [p for p in itertools.combinations(use_ids, 2)]

    def mean_model_point(row_id): 
        if len(key_vids[row_id]) > 1:
            return np.mean(model[key_vids[row_id]].r, axis=0)
        else:
            return model[key_vids[row_id]].r

    dist3d = np.array([np.linalg.norm(mean_model_point(p[0]) - mean_model_point(p[1])) for p in pairs])
    dist2d = np.array([np.linalg.norm(keypoints[p[0], :] - keypoints[p[1], :]) for p in pairs])
    est_ds = focal_length * dist3d / dist2d

    return np.array([0., 0., np.median(est_ds)])


def estimate_global_pose(landmarks, key_vids, model, cam, img, fix_t=False, viz=False, SOLVE_FLATER=True):
    '''
    Estimates the global rotation and translation.
    only diff in estimate_global_pose from single_frame_ferrari is that all animals have the same kp order.
    '''
    # Redefining part names..
    part_names = ['leftEye', 'rightEye', 'chin', 'frontLeftFoot', 'frontRightFoot', 'backLeftFoot', 'backRightFoot', 'tailStart', 'frontLeftKnee', 'frontRightKnee', 'backLeftKnee', 'backRightKnee', 'leftShoulder', 'rightShoulder', 'frontLeftAnkle', 'frontRightAnkle', 'backLeftAnkle', 'backRightAnkle', 'neck', 'TailTip']

    # Use shoulder to "knee"(elbow) distance. also tail to "knee" if available.
    use_names = ['neck', 'leftShoulder', 'rightShoulder', 'backLeftKnee', 'backRightKnee', 'tailStart', 'frontLeftKnee', 'frontRightKnee']
    use_ids = [part_names.index(name) for name in use_names]
    # These might not be visible
    visible = landmarks[:, 2].astype(bool)
    use_ids = [id for id in use_ids if visible[id]]
    if len(use_ids) < 3:
        print('Frontal?..')
        use_names += ['frontLeftAnkle', 'frontRightAnkle', 'backLeftAnkle', 'backRightAnkle']
        model.pose[1] = np.pi/2

    init_t = estimate_translation(landmarks, key_vids, cam.f[0].r, model)

    use_ids = [part_names.index(name) for name in use_names]
    use_ids = [id for id in use_ids if visible[id]]

    # Setup projection error:
    all_vids = np.hstack([key_vids[id] for id in use_ids])
    cam.v = model[all_vids]

    keypoints = landmarks[use_ids, :2].astype(float)

    # Duplicate keypoints for the # of vertices for that kp.
    num_verts_per_kp = [len(key_vids[row_id]) for row_id in use_ids]
    j2d = np.vstack([np.tile(kp, (num_rep, 1)) for kp, num_rep in zip(keypoints, num_verts_per_kp)])

    assert(cam.r.shape == j2d.shape)    

    # SLOW but correct method!!
    # remember which ones belongs together,,
    group = np.hstack([index * np.ones(len(key_vids[row_id])) for index, row_id in enumerate(use_ids)])
    assignments = np.vstack([group == i for i in np.arange(group[-1]+1)])
    num_points = len(use_ids)
    proj_error = (ch.vstack([cam[choice] if np.sum(choice) == 1 else cam[choice].mean(axis=0) for choice in assignments]) - keypoints) / np.sqrt(num_points)

    # Fast but not matching average:
    # Normalization weight
    j2d_norm_weights = np.sqrt(1. / len(use_ids) * np.vstack([1./num * np.ones((num, 1)) for num in num_verts_per_kp]))
    proj_error_fast = j2d_norm_weights * (cam - j2d)

    if fix_t:
        obj = {'cam': proj_error_fast }
    else:
        obj = {'cam': proj_error_fast, 'cam_t': 1e1*(model.trans[2] - init_t[2])}

    # Only estimate body orientation
    if fix_t:
        free_variables = [model.pose[:3]]
    else:
        free_variables = [model.trans, model.pose[:3]]

    if not SOLVE_FLATER:
        obj['feq'] = 1e3 * (cam.f[0] - cam.f[1])
        # So it's under control
        obj['freg'] = 1e1 * (cam.f[0] - 3000) / 1000.
        # here without this cam.f goes negative.. (asking margin of 500)
        obj['fpos'] = ch.maximum(0, 500-cam.f[0])
        # cam t also has to be positive! 
        obj['cam_t_pos'] = ch.maximum(0, 0.01-model.trans[2])
        del obj['cam_t']
        free_variables.append(cam.f)

    if viz:
        import matplotlib.pyplot as plt
        plt.ion()
        def on_step(_):
            plt.figure(1, figsize=(5, 5))
            plt.cla()
            plt.imshow(img[:, :, ::-1])
            img_here = render_mesh(Mesh(model.r, model.f), img.shape[1], img.shape[0], cam)
            plt.imshow(img_here)
            plt.scatter(j2d[:, 0], j2d[:, 1], c='w')
            plt.scatter(cam.r[:, 0], cam.r[:, 1])
            plt.draw()
            plt.pause(1e-3)
            if 'feq' in obj:
                print('flength %.1f %.1f, z %.f' % (cam.f[0], cam.f[1], model.trans[2]))
    else:
        on_step = None

    from time import time
    t0 = time()
    init_angles = [[0,0,0]] #, [1.5,0,0], [1.5,-1.,0]]
    scores = np.zeros(len(init_angles))
    for i,angle in enumerate(init_angles):
        # Init translation
        model.trans[:] = init_t
        model.pose[:3] = angle
        ch.minimize( obj, x0=free_variables, method='dogleg', callback=on_step, options={'maxiter': 100, 'e_3': .0001})
        scores[i] = np.sum(obj['cam'].r**2.)
    j = np.argmin(scores)
    model.trans[:] = init_t
    model.pose[:3] = init_angles[j]
    ch.minimize( obj, x0=free_variables, method='dogleg', callback=on_step, options={'maxiter': 100, 'e_3': .0001})
    
    print('Took %g' % (time() - t0))

    #import pdb; pdb.set_trace()

    if viz:
        dist = np.mean(model.r, axis=0)[2]
        img_here = render_mesh(Mesh(model.r, model.f), img.shape[1], img.shape[0], cam)
        plt.imshow(img[:, :, ::-1])
        plt.imshow(img_here)

    return model.pose[:3].r, model.trans.r

