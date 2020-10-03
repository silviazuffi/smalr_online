'''


'''
import numpy as np
import chumpy as ch
from ARAP import ARAP, wedges, edgesIdx
from chumpy.ch import MatVecMult
from util.myrenderer import render_mesh
from util.image import resize_img
from os.path import join, exists, splitext, basename
from os.path import join, exists, splitext, basename
from os import makedirs
from util.myrenderer import render_mesh, render_orth, trim_sides
import scipy.sparse as sps
import pickle as pkl
import cv2
from smalr_settings import settings, clean_from_green

from silhouette_optimize_smal import fit_silhouettes_pyramid_opt
from set_pose_objs import set_pose_objs

from sbody.texture.utilities import generate_template_map_by_triangles, uv_to_xyz_and_normals
from sbody.texture.mapping import camera_projection
from sbody.laplacian import laplacian
from psbody.mesh.visibility import visibility_compute
from psbody.mesh.meshviewer import MeshViewer, MeshViewers
from psbody.mesh import Mesh
from smpl_webuser.verts import verts_decorated

use_old_camera = False

def load_shape_models(nViews, opt_model_dir, dv, model, frameId, mesh_v_opt_save_path, pose, trans, betas):

    shape_model = [None]*nViews
    v_template = ch.array(model.v_template) + dv

    if exists(mesh_v_opt_save_path):
        opti_mesh = Mesh(filename=mesh_v_opt_save_path)
        v_opt = ch.array(opti_mesh.v.copy())
        compute = False
    else:
        v_opt = None
        compute = True

    for i in range(nViews):
        if v_opt is not None:
            v_template = v_opt
            model.betas[:] = 0
            np_betas = np.zeros_like(model.betas)
        else:
            v_template = ch.array(model.v_template) + dv
            np_betas = np.zeros_like(model.betas)
            np_betas[:len(betas[i])] = betas[i]

        shape_model[i] = verts_decorated(
            v_template=v_template,
            pose=ch.array(pose[i]),
            trans=ch.array(trans[i]),
            J=model.J_regressor,
            kintree_table=model.kintree_table,
            betas=ch.array(np_betas), 
            weights=model.weights,
            posedirs=model.posedirs,
            shapedirs=model.shapedirs,
            bs_type='lrotmin',
            bs_style='lbs',
            f=model.f)

    return shape_model, compute


def optimize_smal(fposes, ftrans, fbetas, model, cams, segs, imgs, landmarks, landmarks_names,
    key_vids, symIdx=None, frameId=0, opt_model_dir=None, save_name=None, 
    COMPUTE_OPT=True, img_paths=None, img_offset=None, img_scales=None):

    mesh_v_opt_save_path = join(opt_model_dir, 'mesh_v_opt_no_mc_'+str(frameId) +'.ply')
    mesh_v_opt_mc_save_path = join(opt_model_dir, 'mesh_v_opt_'+str(frameId) +'.ply')
    mesh_init_save_path = join(opt_model_dir, 'mesh_init_'+str(frameId) +'.ply')
    nViews = len(fposes)
    if not COMPUTE_OPT:
        if not exists(opt_model_dir):
            makedirs(opt_model_dir)
        dv = 0
        compute_texture(nViews, opt_model_dir, dv, model, frameId, mesh_init_save_path,
            fposes, ftrans, fbetas, '_no_refine', cams, imgs, segs, 
            img_paths, img_offset, img_scales)
        return

    # Write the initial mesh
    np_betas = np.zeros_like(model.betas)
    np_betas[:len(fbetas[0])] = fbetas[0]
    tmp = verts_decorated(
            v_template=model.v_template,
            pose=ch.zeros_like(model.pose.r),
            trans=ch.zeros_like(model.trans),
            J=model.J_regressor,
            kintree_table=model.kintree_table,
            betas=ch.array(np_betas),
            weights=model.weights,
            posedirs=model.posedirs,
            shapedirs=model.shapedirs,
            bs_type='lrotmin',
            bs_style='lbs',
            f=model.f)
    tmp_mesh = Mesh(v=tmp.r, f=tmp.f)

    tmp_path = join(opt_model_dir, 'mesh_init_'+str(frameId) +'.ply')
    tmp_mesh.write_ply(tmp_path)
    del tmp

    assert(nViews == len(cams))
    assert(nViews == len(segs))
    assert(nViews == len(imgs))

    # Define a displacement vector. We set a small non zero displacement as initialization
    dv = ch.array(np.random.rand(model.r.shape[0], 3)/1000.)

    # Cell structure for ARAP
    f = model.f
    _, A3, A = edgesIdx(nV=dv.shape[0], f=f, save_dir='.', name='smal')
    wedge = wedges(A3, dv)

    s = np.zeros_like(dv)
    arap = ARAP(reg_e=MatVecMult(A3.T, model.ravel()+dv.ravel()).reshape(-1, 3),
                    model_e=MatVecMult(A3.T, model.ravel()).reshape(-1, 3), w=wedge, A=A)

    k_arap = settings['ref_k_arap_per_view']*nViews 
    for weight, part in zip(settings['ref_W_arap_values'], settings['ref_W_arap_parts']):
        k_arap, W_per_vertex = get_arap_part_weights(A, k_arap, [part], [weight]) #, animal_name) # was only Head

    W = np.zeros((W_per_vertex.shape[0],3))
    for i in range(3):
        W[:,i] = W_per_vertex

    k_lap = settings['ref_k_lap']*nViews*W
    k_sym = settings['ref_k_sym']*nViews
    k_keyp = settings['ref_k_keyp_weight']*nViews

    # Load already computed mesh
    if not exists(opt_model_dir):
        makedirs(opt_model_dir)
    shape_model, compute = load_shape_models(nViews, opt_model_dir, dv, model, frameId, mesh_v_opt_save_path, fposes, ftrans, fbetas)

    mv = None

    # Remove inside mouth faces
    '''
    if settings['ref_remove_inside_mouth']:
        # Giraffe
        faces_orig = shape_model[0].f.copy()
        im_v = im_up_v + im_down_v
        idx = [np.where(model.f == ix)[0] for ix in im_v]
        idx = np.concatenate(idx).ravel()
        for i in range(nViews):
            shape_model[i].f = np.delete(shape_model[i].f, idx, 0)
    '''

    if compute:
        objs = {}
  
        FIX_CAM = True
        free_variables = []
        kp_weights = k_keyp*np.ones((landmarks[0].shape[0],1))
        print('removing shoulders, often bad annotated')
        kp_weights[landmarks_names.index('leftShoulder'),:] *= 0
        kp_weights[landmarks_names.index('rightShoulder'),:] *= 0
        objs_pose = None
        j2d = None
        #k_silh_term = settings['ref_k_silh_term']  
        k_m2s = settings['ref_k_m2s']    
        k_s2m = settings['ref_k_s2m']   

        objs, params_, j2d = set_pose_objs(shape_model, cams, landmarks, key_vids, kp_weights=kp_weights, FIX_CAM=FIX_CAM, ONLY_KEYP=True, OPT_SHAPE=False)

        if np.any(k_arap) != 0:
            objs['arap'] = k_arap*arap 
        if k_sym != 0:
            objs['sym_0'] = k_sym*(ch.abs(dv[:,0] - dv[symIdx,0])) 
            objs['sym_1'] = k_sym*(ch.abs(dv[:,1] + dv[symIdx,1] - 0.00014954)) 
            objs['sym_2'] = k_sym*(ch.abs(dv[:,2] - dv[symIdx,2])) 
        if np.any(k_lap) != 0:
            lap_op = np.asarray(laplacian(Mesh(v=dv,f=shape_model[0].f)).todense())
            objs['lap'] = k_lap*ch.dot(lap_op, dv)

        mv = None
        mv2 = MeshViewers(shape=(1,nViews)) #None
        vc = np.ones_like(dv)
        dv_r = fit_silhouettes_pyramid_opt(objs,
                            shape_model, dv,
                            segs,
                            cams,
                            j2d=j2d,
                            weights=1.,
                            mv=mv,
                            imgs=imgs,
                            s2m_weights=k_s2m,
                            m2s_weights=k_m2s,
                            max_iter=100,
                            free_variables=free_variables,
                            vc=vc, symIdx=symIdx, mv2=mv2, objs_pose=objs_pose)

        # Save result image
        for i in range(nViews):
            img_res = render_mesh(Mesh(shape_model[i].r, shape_model[i].f),
                      imgs[i].shape[1], imgs[i].shape[0], cams[i], img=imgs[i],  world_frame=True)
            img_result = np.hstack((imgs[i], img_res * 255.))
            save_img_path = save_name[i].replace('.pkl', '_v_opt.png')
            cv2.imwrite(save_img_path, img_result)

        shape_model[0].pose[:] = 0
        shape_model[0].trans[:] = 0
        V = shape_model[0].r.copy()
        vm = V[symIdx,:].copy()
        vm[:,1] = -1*vm[:,1]
        V2 = (V+vm)/2.0

        mesh_out = Mesh(v=V2, f=shape_model[0].f)
        mesh_out.show()
        mesh_out.write_ply(mesh_v_opt_save_path)

        save_dv_data_path = mesh_v_opt_save_path.replace('.ply', '_dv.pkl')
        dv_data = {'betas':shape_model[0].betas.r, 'dv':dv_r}
        pkl.dump(dv_data, open(save_dv_data_path, 'wb'))

        
    compute_texture(nViews, opt_model_dir, dv, model, frameId, mesh_v_opt_save_path,
                    fposes, ftrans, fbetas, '_non_opt', cams, imgs, segs,
                    img_paths, img_offset, img_scales)

    return

def compute_texture(nViews, opt_model_dir, dv, model, frameId, mesh_v_opt_save_path,
                    fposes, ftrans, fbetas, code, cams, imgs, segs, 
                    img_paths, img_offset, img_scales):

    # If the images have been scaled, use the original images
    for i in range(nViews):
        if img_scales[i] < 1.0:
            imgs[i] = cv2.imread(img_paths[i])
            border = int(img_offset[i]/img_scales[i])
            w, h, d = imgs[i].shape
            tmp = np.zeros((w+2*border, h+2*border, d), dtype=np.uint8)
            tmp[border:border+w, border:border+h, :] = imgs[i]
            imgs[i] = tmp.copy()
            segs[i] = cv2.resize(segs[i], (imgs[i].shape[1], imgs[i].shape[0]))
            try:
                assert(imgs[i].shape == segs[i].shape)
            except:
                import pdb; pdb.set_trace()

            cams[i].f[:] = cams[i].f[:].r/img_scales[i]
            cams[i].c[0] = np.array(imgs[i].shape[1])/2.0
            cams[i].c[1] = np.array(imgs[i].shape[0])/2.0


    shape_model, _ = load_shape_models(nViews, opt_model_dir, dv, model, frameId, mesh_v_opt_save_path, fposes, ftrans, fbetas)

    # Load the texture coordinates
    uv_mesh = Mesh(filename=settings['template_w_tex_uv_name'])

    # Set of aligned meshes
    algn = [Mesh(v=mod.r, f=mod.f) for mod in shape_model]

    # Add the texture coo to the meshes and to a mesh in t-pose
    tmpl = Mesh(v=shape_model[0].v_template.r, f=shape_model[0].f)
    for i in range(nViews):
        algn[i].ft = uv_mesh.ft 
        algn[i].vt = uv_mesh.vt 
    tmpl.ft = uv_mesh.ft 
    tmpl.vt = uv_mesh.vt 

    # Load the information for faces symmetry
    data = pkl.load(open('symmetry_indexes.pkl', 'rb'), encoding='latin1')

    fSymIdx = data['fSymIdx']

    mask_colored = cv2.imread(settings['texture_map_colored_name']) #'texture_mask_colored.png')
    color_locations_w = settings['texture_color_locations'][0]
    color_locations_h = settings['texture_color_locations'][1]

    # Get texture
    scale = mask_colored.shape[0]/2048.
    texture_path = join(opt_model_dir, 'texture_'+str(frameId)+code+'.png')

    (face_indices_map, b_coords_map) = generate_template_map_by_triangles(tmpl, map_scale=scale)
    full_texture, sum_of_weights = my_color_map_by_proj(shape_model, algn, cams, face_indices_map, b_coords_map,
        source_images=imgs, silhs=segs, save_path=texture_path)

    # Generate a symmetrized texture map
    tmpl_s = Mesh(v=shape_model[0].v_template.r, f=shape_model[0].f)
    uv_f = uv_mesh.ft[fSymIdx,:].copy()
    uv_f[:,[0,1,2]] = uv_f[:,[0,2,1]]
    tmpl_s.ft = uv_f 
    uv_s = uv_mesh.vt.copy()
    tmpl_s.vt = uv_s 

    # Get texture for the symmetric mesh
    scale = mask_colored.shape[0]/2048.
    texture_path_s = join(opt_model_dir, 'texture_sym_'+str(frameId)+code+'.png')
    (face_indices_map, b_coords_map) = generate_template_map_by_triangles(tmpl_s, map_scale=scale)
    full_texture_s, sum_of_weights_s = my_color_map_by_proj(shape_model, algn, cams, face_indices_map, b_coords_map,
        source_images=imgs, silhs=segs, save_path=texture_path_s)

    W = sum_of_weights.copy()
    W[W<=.00001] = 0.
    W_s = sum_of_weights_s.copy()
    W_s[W_s<=.00001] = 0.

    # Get the average animal color to fill-in non covered areas
    a_color = np.zeros((3))
    for i in range(3):
        a_color[i] = np.median(full_texture[W>0,i])
    
    # Take the average
    print('computing average texture')
    texture_final_avg = a_color*np.ones_like(full_texture)
    W_both = W+W_s
    idx = np.where(W_both>0)
    for j in range(full_texture.shape[2]):
        texture_final_avg[:,:,j][idx] = (full_texture[:,:,j][idx]*W[idx] + full_texture_s[:,:,j][idx]*W_s[idx])/W_both[idx]

    # Fill-in the parts that have not been assigned with the average color
    for x,y in zip(color_locations_w, color_locations_h):
        col = mask_colored[y,x,:]
        idx0 = (mask_colored[:,:,0]==col[0])
        idx1 = (mask_colored[:,:,1]==col[1])
        idx2 = (mask_colored[:,:,2]==col[2])
        idx = idx0*idx1*idx2
        M = np.zeros_like(W_both)
        M[idx] = 1.
        a_col = np.zeros((3))
        idx = (W_both>0)*M>0
        a_col[0] = np.median(texture_final_avg[idx,0])
        a_col[1] = np.median(texture_final_avg[idx,1])
        a_col[2] = np.median(texture_final_avg[idx,2])
        idx = (W_both==0)*M>0
        texture_final_avg[idx,:] = a_col
        
    texture_path_f_avg = join(opt_model_dir, 'texture_final_average_'+str(frameId)+code+'.png')
    cv2.imwrite(texture_path_f_avg, texture_final_avg*255)
    
    print('filling in texture')
    texture_final = full_texture.copy()
    # Fill-in if not present
    texture_final[W==0,0] = texture_final_avg[W==0,0] 
    texture_final[W==0,1] = texture_final_avg[W==0,1] 
    texture_final[W==0,2] = texture_final_avg[W==0,2] 
    texture_path_f = join(opt_model_dir, 'texture_final_filled_'+str(frameId)+code+'.png')
    cv2.imwrite(texture_path_f, texture_final*255)

    return

def old_camera(pp_cam, w, h):
    from sbody.calib.camera import Camera
    cam = Camera(size=(w, h), k=pp_cam.k.reshape((-1,1)), r=pp_cam.rt.reshape((-1,1)), t=pp_cam.t.reshape((-1,1)),\
            f=pp_cam.f.reshape((-1,1)), c=pp_cam.c.reshape((-1,1)))
    return cam

def my_color_map_by_proj(svs, algn, cams, face_indices_map, b_coords_map, source_images=None, silhs=None, save_path='texture.png'):

    nCams = len(cams)
    texture_maps = []
    weights = []
    vis = [None]*nCams
    for i in range(nCams):

        print("working on camera %d" % i)
        alignment = Mesh(v=algn[i].v, f=algn[i].f)
        (points, normals) = uv_to_xyz_and_normals(alignment, face_indices_map, b_coords_map)

        # add artificious vertices and normals
        alignment.points = points
        alignment.v = np.vstack((alignment.v, points))
        alignment.vn = np.vstack((alignment.vn, normals))

        img = source_images[i]
        if use_old_camera:
            camera = old_camera(cams[i], img.shape[1], img.shape[0])
       
            cam_vis_ndot = np.array(visibility_compute(v=alignment.v, f=alignment.f, n=alignment.vn, \
                cams=(np.array([camera.origin.flatten()])))) 
        else:
            camera = cams[i]
            cams[i].v = points
            cam_vis_ndot = np.array(visibility_compute(v=alignment.v, f=alignment.f, n=alignment.vn, \
                cams=(np.array([camera.t.r.flatten()])))) 

        cam_vis_ndot = cam_vis_ndot[:,0,:]
        (cmap, vmap) = camera_projection(alignment, camera, cam_vis_ndot, img, face_indices_map, b_coords_map)

        vis[i] = cam_vis_ndot[0][-alignment.points.shape[0]:]
        n_dot = cam_vis_ndot[1][-alignment.points.shape[0]:]
        vis[i][n_dot<0] = 0
        n_dot[n_dot<0] = 0
        texture_maps.append(cmap)
        weights.append(vmap)

        imgf = render_mesh(alignment, img.shape[1],
            img.shape[0], cams[i], img=img, world_frame=True)

        cv2.imwrite('texture_'+str(i)+'.png', texture_maps[i])
        cv2.imwrite('texture_w_'+str(i)+'.png', 255*vmap)

        # restore old vertices and normals
        alignment.v = alignment.v[:(len(alignment.v)-points.shape[0])]
        alignment.vn = alignment.vn[:(len(alignment.vn)-points.shape[0])]
        del alignment.points

    # Create a global texture map
    # Federica Bogo's code
    sum_of_weights = np.array(weights).sum(axis=0)
    sum_of_weights[sum_of_weights == 0] = .00001
    for weight in weights:
        weight /= sum_of_weights

    if settings['max_tex_weight']:
        W = np.asarray(weights)
        M = np.max(W, axis=0)
        for i in range(len(weights)):
            B = weights[i]!=0
            weights[i] = (W[i,:,:]==M)*B

    if clean_from_green:
        print('Cleaning green pixels')
        weights_green = clean_green(texture_maps, source_images, silhs)
        weights_all = [weights_green[i]*w for i,w in enumerate(weights)]
    else:
        weights_all = weights

    sum_of_weights = np.array(weights_all).sum(axis=0)
    if not settings['max_tex_weight']:
        sum_of_weights[sum_of_weights == 0] = .00001
        for w in weights_all:
            w /= sum_of_weights
    
    full_texture_med = np.median(np.array([texture_maps]), axis=1).squeeze()/255.
    T = np.array([texture_maps]).squeeze()/255.
    W = np.zeros_like(T)
    if nCams > 1:
        W[:,:,:,0] = np.array([weights_all]).squeeze()
        W[:,:,:,1] = np.array([weights_all]).squeeze()
        W[:,:,:,2] = np.array([weights_all]).squeeze()
    else:
        W[:,:,0] = np.array([weights_all]).squeeze()
        W[:,:,1] = np.array([weights_all]).squeeze()
        W[:,:,2] = np.array([weights_all]).squeeze()

    # Average texture
    for i, texture in enumerate(texture_maps):
        for j in range(texture.shape[2]):
            texture[:,:,j] = weights_all[i]*texture[:,:,j]/255.
    full_texture = np.sum(np.array([texture_maps]), axis=1).squeeze()
    cv2.imwrite(save_path, full_texture*255)
    return full_texture, sum_of_weights
    
def get_arap_part_weights(A, k_arap, parts, partWeights): 

    data = pkl.load(open('gloss_data.pkl', 'rb'), encoding='latin1')
    partSet = data['partSet']
    model_parts = data['model_parts']
    model_part2bodyPoints = data['model_part2bodyPoints']

    W = np.ones(A.shape[0])
    W_arap = np.ones(len(partSet))
    for i, part in enumerate(parts):
        W_arap[model_parts[part]] = partWeights[i] 
    for i in partSet:
        pidx = model_part2bodyPoints[i]
        W[pidx] = W_arap[i]
    aweight = np.zeros(3*A.shape[1])
    I = sps.find(A)
    for j in range(len(I[0])):
        if I[2][j] > 0:
            aweight[3*I[1][j]:3*I[1][j]+3] = W[I[0][j]]
    k_arap = k_arap*aweight
    return k_arap, W

def colorate_template_map_by_segm(template, segm, scale):

    texture_image = np.zeros((2048, 2048))       
    map_height = np.int32(texture_image.shape[0]/scale)
    map_width = np.int32(texture_image.shape[1]/scale)
    face_indices_map = np.zeros((map_height, map_width, 3))

    text_coords = template.vt[:,:2]
    text_coords[:,0] *= map_width
    text_coords[:,1] = (1-text_coords[:,1])*map_height
    text_coords = np.int32(text_coords)

    parts = range(1+np.max(segm))
    for part in parts:
            indices = np.where(segm==part)[0]
            color = np.int32(np.random.rand(3)*255)
            while np.all(color == [0,0,0]):
                    color = np.int32(np.random.rand(3)*255)
            for face in indices:
                    points = text_coords[template.ft[face]]
                    points = np.int32([points]).squeeze() 
                    cv2.fillConvexPoly(face_indices_map, points, color.tolist())

    b_coords_map = face_indices_map
    unmapped = (np.sum((b_coords_map.reshape(b_coords_map.shape[0]*b_coords_map.shape[1],3) == -np.zeros(3)),axis=1) == 3).reshape(b_coords_map.shape[0], b_coords_map.shape[1])
    dst, lab = cv2.distanceTransformWithLabels(unmapped.astype(np.uint8), distanceType=1, maskSize=3, labelType=1)
    full_texture = face_indices_map[np.logical_not(unmapped)][lab-1]
    return full_texture


def clean_green(textures, img_orig, seg_orig):

    # Add the black pixels to the mask
    mask = [None]*len(textures)
    for i,seg in enumerate(seg_orig):
        mask[i] = np.zeros_like(seg)
        mask[i][seg>0] = 255
        lab = cv2.cvtColor(np.uint8(img_orig[i]), cv2.COLOR_BGR2LAB)
        mask[i][lab[:,:,0]<10.0] = 255

    # Get the average green pixels in each view
    hsv = [cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV) for img in img_orig]
    green_hsv = [np.mean(img[mask[i][:,:,0]==0], axis=0) for i,img in enumerate(hsv)]
    green_rgb = [np.mean(img[mask[i][:,:,0]==0], axis=0) for i,img in enumerate(img_orig)]

    texs_hsv = [cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV) for img in textures]
    dist2green = [np.abs(tex[:,:,0] - green_hsv[i][0]) for i,tex in enumerate(texs_hsv)]
    dist2green2 = [np.abs(tex[:,:,0] - 40) for i,tex in enumerate(texs_hsv)]

    # Convert distences in weights for the texture compositing
    weights = [None]*len(textures)
    for i,dist in enumerate(dist2green):
        weights[i] = np.ones_like(dist)
        weights[i][dist<20] = 0.
        weights[i][dist2green2[i]<10] = 0.
        #pylab.figure()
        #pylab.subplot(1,2,1)
        #pylab.imshow(weights[i])
        #pylab.subplot(1,2,2)
        #pylab.imshow(dist)
        #import pdb; pdb.set_trace()

    return weights

