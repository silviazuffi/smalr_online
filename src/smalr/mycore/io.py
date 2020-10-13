def load_seg(img_path, return_path=False, img_ext='.png'):
    from os.path import basename, dirname, join, splitext
    import cv2
    seg_path = join(dirname(img_path), 'bgsub', basename(img_path))
    seg_path = splitext(seg_path)[0]+img_ext
    if not return_path:
        return cv2.imread(seg_path)
    else:
        return cv2.imread(seg_path), seg_path

def get_anno_path(img_path, post_fix='_smpl.mat'):
    from os.path import join, dirname, basename

    ext = ('.jpg', '.JPG', '.png')
    for ex in ext:
        if ex in img_path:
            basename0 = basename(img_path).replace(ex, post_fix)

    anno_path = join(dirname(img_path), 'annotations', basename0)

    return anno_path


def load_keypoints(anno_path, ret_all=False):
    if '.png' in anno_path:
        anno_path = get_anno_path(anno_path)
    import scipy.io as sio

    res = sio.loadmat(anno_path, squeeze_me=True, struct_as_record=False)
    res = res['annotation']
    j2d = res.twoD.astype('float') - 1
    vids = res.v_ids - 1
    if ret_all:
        return (j2d, vids, res)
    else:
        return (j2d, vids)


def load_animal_model(model_name='my_smpl_15.pkl'):
    from smpl_webuser.serialization import load_model
    from os.path import exists, join

    model_dir = '../../'
    model_path = join(model_dir, 'smpl_models', model_name)
    model = load_model(model_path)
    return model


def load_keymapping(animal):
    import scipy.io as sio
    import numpy as np
    from os.path import exists
    map_path = '../../src/annotate_kp_matlab/ferrari2smpl_%s.mat' % animal
    mapping = sio.loadmat(map_path, squeeze_me=True, struct_as_record=False)['map']
    v_ids = mapping.vids
    # Make single keyids into array instead of bare int..
    # -1 because the indices are in matlab.
    v_ids = np.array([np.atleast_1d(v) - 1 for v in v_ids])

    kp_names = mapping.names
    kp_names = [name.encode('ascii', 'ignore') for name in kp_names]
    return v_ids, kp_names

def crop_img(img, landmarks, min_size=300, max_width=1000, seg=None, get_rect=False):

    vis = landmarks[:, 2].astype(bool)
    kp = landmarks[:, :2].astype(float)
    # Crop if the animal is too small..
    # Get bbox:
    img_h, img_w = img.shape[:2]
    # margin = 80
    import numpy as np
    margin = np.array([0.2, 0.13]) * (img_h + img_w) / 2.
    ymin = min(kp[vis, 1])
    ymax = max(kp[vis, 1])
    xmin = min(kp[vis, 0])
    xmax = max(kp[vis, 0])

    if seg is not None: # Use seg too.
        rows, cols = np.nonzero(seg)
        ymin = min(min(rows), ymin)
        ymax = max(max(rows), ymax)
        xmin = min(min(cols), xmin)
        xmax = max(max(cols), xmax)

    ymin = max(0, ymin - margin[1])
    ymax = min(img_h - 1, ymax + margin[1])
    xmin = max(0, xmin - margin[0])
    xmax = min(img_w - 1, xmax + margin[0])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    rect = (xmin, ymin, w, h)

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle
    # plt.ion()
    # plt.clf()
    # ax = plt.subplot(121)
    # plt.imshow(img[:, :, ::-1])
    # plt.scatter(kp[vis, 0], kp[vis, 1])
    # ax.add_patch(Rectangle((xmin, ymin), w, h, alpha=1, facecolor='none'))
    # plt.draw()
    # import ipdb; ipdb.set_trace()

    # Crop if it doesn't occupy much of image:
    import numpy as np
    if w < 0.9* img_w and h < 0.9*img_h:
        new_img = img[ymin:ymax, xmin:xmax, :]
        new_kp = kp
        new_kp[vis] = kp[vis] - np.array([xmin, ymin])
        if seg is not None:
            seg = seg[ymin:ymax, xmin:xmax]
    elif w < 0.9* img_w: # Just the sides.
        new_img = img[:, xmin:xmax, :]
        new_kp = kp
        new_kp[vis] = kp[vis] - np.array([xmin, 0])
        if seg is not None:
            seg = seg[:, xmin:xmax]
    elif h < 0.9* img_h: # Just the top.
        new_img = img[ymin:ymax, :, :]
        new_kp = kp
        new_kp[vis] = kp[vis] - np.array([0, ymin])
        if seg is not None:
            seg = seg[ymin:ymax, :]
    else:
        new_img = img
        new_kp = kp

    # plt.subplot(122)
    # plt.imshow(new_img[:, :, ::-1])
    # plt.scatter(new_kp[vis, 0], new_kp[vis, 1])

    # Scale if its too small.
    import cv2
    if max(new_img.shape[:2]) < min_size:
        scale_factor = float(min_size) / max(new_img.shape[:2])
        new_size = (np.array(new_img.shape[:2]) * scale_factor).astype(int)
        new_img = cv2.resize(new_img, (new_size[1], new_size[0]))
        new_kp *= scale_factor
        if seg is not None:
            seg = cv2.resize(seg, (new_size[1], new_size[0]))
    elif new_img.shape[1] > max_width:
        scale_factor = float(max_width) / new_img.shape[1]
        new_size = (np.array(new_img.shape[:2]) * scale_factor).astype(int)

        new_img = cv2.resize(new_img, (new_size[1], new_size[0]))
        new_kp *= scale_factor
        if seg is not None:
            seg = cv2.resize(seg, (new_size[1], new_size[0]))

    # plt.subplot(122)
    # plt.imshow(new_img[:, :, ::-1])
    # plt.scatter(new_kp[vis, 0], new_kp[vis, 1])
    # import ipdb; ipdb.set_trace()
    new_landmarks = np.hstack([new_kp, np.atleast_2d(vis).T])

    if get_rect:
        return new_img, new_landmarks, seg, rect
    else:
        return new_img, new_landmarks, seg

def read_image(img_path):
    import cv2
    from os.path import exists, splitext
    # Check if the segmentation exists Or if the image has alpha channel.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        # Grayscale! Read it as 3D.
        img = cv2.imread(img_path)
        seg_path = splitext(img_path)[0] + '_s.png'
        if exists(seg_path):
            alpha = cv2.imread(seg_path)
            alpha = (alpha[:, :, 0] > 150.).astype('uint8')
        else:
            alpha = None
    elif img.shape[2] == 4:
        alpha = (img[:, :, 3] > 150.).astype('uint8')
        b, g, r = cv2.split(img[:, :, :3])
        b[alpha == 0] = 255
        g[alpha == 0] = 255
        r[alpha == 0] = 255
        img = cv2.merge((b,g,r))
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.imshow(img[:, :, ::-1])
        # import ipdb; ipdb.set_trace()
    #else:
    seg_path = splitext(img_path)[0] + '_s.png'
    if exists(seg_path):
        alpha = cv2.imread(seg_path)
        alpha = (alpha[:, :, 0] > 150.).astype('uint8')
    else:
        alpha = None

    return img, alpha
