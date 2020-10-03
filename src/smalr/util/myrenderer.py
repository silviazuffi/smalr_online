'''
My renderer utils in one place.
Written by: Angjoo K.
Modified by Silvia Z.
'''
import numpy as np
import cv2
from opendr.camera import ProjectPoints
from psbody.mesh import Mesh

colors = {
    # 'pink': [.7, .7, .9],
    # 'neutral': [.9, .9, .8],
    # 'capsule': [.7, .75, .5],
    # 'yellow': [.5, .7, .75],
    # Reversing these bc opencv writes BGR
    # gray:
    'dog'   : np.array([0.8   , 0.76, 0.77  ])[::-1],
    # yellow-green
    # 'cow' : np.array([0.671   , 0.741, 0.369  ])[::-1],
    'blue'   : np.array([0.55  , 0.71, 0.8   ])[::-1],
    # mint
    'others'   : np.array([0.59  , 0.8,  0.8   ])[::-1],
    # pink
    # 'others'   : np.array([0.8 , 0.55, 0.58  ])[::-1],
    'cow'   : np.array([0.933 , 0.643, 0.6784  ])[::-1],
    # lilac:
    'hippo'  : np.array([0.851 , .729, 0.894 ])[::-1],
    # beige:
    'horse'  : np.array([0.8   , .73, 0.59 ])[::-1],
    # lblue:
    'big_cats'  : np.array([0.718 , .824, 0.988 ])[::-1],
    # 'big_cats'  : np.array([.8314 , .6863, .2157 ])[::-1],
    # gold
    'gold'  : np.array([0.718 , .824, 0.988 ]),
}
def world_to_cam(Pw, camera):
    R,_ = cv2.Rodrigues(camera.rt.r)
    P = Pw.dot(R.T)+camera.t.r
    return P

# --------------------Rendering stuff --------------------
def create_renderer(w=640,
                    h=480,
                    rt=np.zeros(3),
                    t=np.zeros(3),
                    f=None,
                    c=None,
                    k=None,
                    near=.5,
                    far=10.,
                    mesh=None):
    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    if mesh is not None and hasattr(mesh, 'texture_image'):
        from opendr.renderer import TexturedRenderer
        rn = TexturedRenderer()
        rn.texture_image = mesh.texture_image
        if rn.texture_image.max() > 1:
            rn.texture_image[:] = rn.texture_image[:].r/255.
        rn.ft = mesh.ft
        rn.vt = mesh.vt
    else:
        from opendr.renderer import ColoredRenderer
        rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def stack_with(rn, mesh):
    if hasattr(rn, 'texture_image'):
        if not hasattr(mesh, 'ft'):
            mesh.ft = mesh.f
            mesh.vt = mesh.v[:, :2]
        rn.ft = np.vstack((rn.ft, mesh.ft + len(rn.vt)))
        rn.vt = np.vstack((rn.vt, mesh.vt))
    rn.f = np.vstack((rn.f, mesh.f + len(rn.v)))
    rn.v = np.vstack((rn.v, mesh.v))
    rn.vc = np.vstack((rn.vc, mesh.vc))


def points_to_spheres(points, radius=0.01, color=[.5, .5, .5]):
    from body.mesh.sphere import Sphere
    spheres = Mesh(v=[], f=[])
    for center in points:
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=color))
    return spheres


def simple_renderer(rn, meshes, yrot=np.radians(120)):
    from opendr.lighting import LambertianPointLight
    mesh = meshes[0]
    if hasattr(rn, 'texture_image'):
        if not hasattr(mesh, 'ft'):
            mesh.ft = copy(mesh.f)
            vt = copy(mesh.v[:, :2])
            vt -= np.min(vt, axis=0).reshape((1, -1))
            vt /= np.max(vt, axis=0).reshape((1, -1))
            mesh.vt = vt
        # mesh.texture_filepath = rn.texture_image
        rn.set(v=mesh.v,
               f=mesh.f,
               vc=mesh.vc,
               ft=mesh.ft,
               vt=mesh.vt,
               bgcolor=np.ones(3))
    else:
        rn.set(v=mesh.v, f=mesh.f, vc=mesh.vc, bgcolor=np.ones(3))

    for next_mesh in meshes[1:]:
        stack_with(rn, next_mesh)

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)
    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def remove_whitespace(fname):
    im = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    imt = np.sum(im / 255., axis=2)
    v1 = np.var(imt, axis=0)
    v0 = np.var(imt, axis=1)

    im = im[v0 != 0, :, :]
    im = im[:, v1 != 0, :]

    cv2.imwrite(fname, im)

def render_meshes(meshes, cam, near=0.5, far=20, img=None, w=None, h=None):
    # Near anre far are depreciated! bc they're calculated in here. 

    if w is None:
        w = img.shape[1]
        h = img.shape[0]

    min_z = 1e10
    max_z = -1e10
    for i in range(len(meshes)):
        mi_z = np.maximum(np.min(meshes[i].v, axis=0)[2] - 5, 0.5)
        ma_z = np.max(meshes[i].v, axis=0)[2] + 5
        if mi_z < min_z:
            min_z = mi_z
        if ma_z > max_z:
            max_z = ma_z
    rn = create_renderer(
        w=w, h=h, near=min_z, far=max_z, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c, k=cam.k)

    if img is not None:
        rn.background_image = img / 255.

    imtmp = simple_renderer(rn=rn, meshes=meshes)
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp


# Main:
def render_mesh(mesh, w, h, cam, near=0.5, far=20, img=None, deg=None, margin=None, color_key='blue', world_frame=False):
    # Near anre far are depreciated! bc they're calculated in here. 
    if margin is not None:
        # Margin is tuple (height, width)
        w = w + 2*margin[1]
        h = h + 2*margin[0]
        orig_c = cam.c.r.copy()
        cam.c += margin[::-1]

        if img is not None:
            # Pad with white
            img = np.pad(img, ((margin[0], margin[0]),(margin[1], margin[1]),(0,0)), 'constant', constant_values=255)

    if world_frame:
        #min_z = cam.t[2]+np.min(mesh.v, axis=0)[2]-5.0
        #max_z = cam.t[2]+np.max(mesh.v, axis=0)[2]+5.0
        min_z = np.maximum(np.min(world_to_cam(mesh.v, cam), axis=0)[2] - 5, 0.5)
        max_z = np.max(world_to_cam(mesh.v, cam), axis=0)[2]
    else:
        min_z = np.maximum(np.min(mesh.v, axis=0)[2] - 5, 0.5)
        max_z = np.max(mesh.v, axis=0)[2] + 5

    rn = create_renderer(
        w=w, h=h, near=min_z, far=max_z, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c, k=cam.k, mesh=mesh)

    if img is not None:
        rn.background_image = img / 255.

    if not hasattr(mesh, 'vc'):
        mesh.vc = colors[color_key]

    if deg is not None:
        import math
        orig_v = mesh.v.copy()
        aroundy = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        center = orig_v.mean(axis=0)
        new_v = np.dot((orig_v - center), aroundy)
        mesh.v = new_v + center

    imtmp = simple_renderer(rn=rn, meshes=[mesh])

    if img is None:
        imtmp = get_alpha(imtmp)

    # Recover original mesh.
    if deg is not None:
        mesh.v = orig_v

    # Recover original
    if margin is not None:
        cam.c = orig_c

    return imtmp

def render_orth(sv, w, h, cam, img=None, deg=None, margin=None, color_key='blue', use_face=None, vc=None):
    # Use large focal length to remove perspective effects.
    # Get projected points using current f.
    orig_trans = sv.trans.r.copy()
    # import ipdb; ipdb.set_trace()
    use_f = 5000.
    # We have f * X/Z = x = f' * X / Z' where f' is the use_f, solve for Z'
    new_Z = use_f / cam.f[0].r * orig_trans[2]
    sv.trans[2] = new_Z
    # Now project this
    # dist = np.mean(sv.r, axis=0)[2]
    min_z = np.maximum(np.min(sv.r, axis=0)[2] - 5, 0.5)
    max_z = np.max(sv.r, axis=0)[2]
    cam_f = ProjectPoints(rt=cam.rt, t=cam.t, f=np.array([use_f, use_f]), c=cam.c, k=cam.k)
    if use_face is None:
        #use_face = sv.model.f
        use_face = sv.f

    if vc is None:
        img_here = render_mesh(Mesh(sv.r, use_face), w, h, cam_f, near=min_z, far=max_z+5, img=img, deg=deg, margin=margin, color_key=color_key)
    else:
        img_here = render_mesh(Mesh(sv.r, use_face, vc=vc), w, h, cam_f, near=min_z, far=max_z+5, img=img, deg=deg, margin=margin, color_key=color_key)

    # Model and sv don't have ties but for safety put it back.
    sv.trans[:] = orig_trans

    return img_here

def render_tpose(sv, w, h, deg=None, margin=None, color_key='blue'):
    # Margin is (height, width)
    orig_pose = sv.pose.r.copy()
    orig_trans = sv.trans.r.copy()
    # sv.pose[3:] = np.zeros_like(sv.pose.r[3:])
    # Default camera:
    sv.pose[:] = np.zeros_like(sv.pose.r)
    sv.pose[:2] = [np.pi/2., 0.1]
    sv.trans[:] = np.array([1.2, -1.4, 33])
    # Custom camera for this default setting.
    from mycore.camera import setup_camera
    cam = setup_camera(600, 300,flength=5000.)

    img_here = render_mesh(Mesh(sv.r, sv.model.f), 600, 300, cam, deg=deg, color_key=color_key)    

    # Now resize so that.. height is the same? 
    img_here = trim_sides(img_here)
    target_h = h if margin is None else h + margin[0] * 2
    scale_factor = float(target_h) / img_here.shape[0]
    new_size = (np.round(np.array(img_here.shape[:2]) * scale_factor)).astype(int)
    img_here0 = cv2.resize(img_here, (new_size[1], new_size[0]))

    sv.pose[:] = orig_pose
    sv.trans[:] = orig_trans

    return img_here0

def trim_sides(imgA):
    alpha = imgA[:, :, 3]
    y, x = np.nonzero(alpha.astype(bool))

    img_h, img_w = imgA.shape[:2]
    margin = 5
    # ymin = max(0, min(y) - margin)
    # ymax = min(img_h - 1, max(y) + margin)
    xmin = max(0, min(x) - margin)
    xmax = min(img_w - 1, max(x) + margin)

    return imgA[:, xmin:xmax]


def trim_tightbox(imgA, just_box=False):
    alpha = imgA[:, :, 3]
    y, x = np.nonzero(alpha.astype(bool))

    img_h, img_w = imgA.shape[:2]
    margin = 0
    ymin = max(0, min(y) - margin)
    ymax = min(img_h - 1, max(y) + margin)
    xmin = max(0, min(x) - margin)
    xmax = min(img_w - 1, max(x) + margin)

    if just_box:
        return np.array([xmin, ymin, xmax, ymax])
    else:
        return imgA[ymin:ymax, xmin:xmax]
