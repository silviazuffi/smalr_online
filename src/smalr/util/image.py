import cv2
from psbody.mesh.colors import name_to_rgb
import numpy as np
from chumpy import array
from opendr.camera import ProjectPoints

def plot_points(points_, img, points0_=None, colors=None, radius=15):
    tmp_img = img.copy()
    if colors is None:
        color_names = ['light cyan', 'HotPink1', 'yellow', 'light blue',
                       'aquamarine', 'SkyBlue', 'magenta3', 'LightGreen',
                       'MistyRose4', 'violet', 'SpringGreen1', 'light pink',
                       'light gray', 'dark olive green', 'rosy brown',
                       'dark cyan', 'maroon', 'IndianRed', 'sea green',
                       'MediumBlue', 'gold1', 'tomato2', 'medium purple',
                       'navy', 'LightPink3', 'PeachPuff1']
        colors = [(name_to_rgb[c] * 255).astype(int) for c in color_names]

    num_colors = len(colors)

    points = points_.T if points_.shape[0] == 2 else points_
    points0 = None if points0_ is None else (points0_.T
                                             if points0_.shape[0] == 2 else
                                             points0_)

    for i, coord in enumerate(points.astype(int)):
        if coord[0] < img.shape[1] and coord[1] < img.shape[0] and coord[
                0] > 0 and coord[1] > 0:
            cv2.circle(tmp_img, tuple(coord), radius,
                       colors[i % num_colors].tolist(), -1)

    if points0 is not None:
        for i, coord in enumerate(points0.astype(int)):
            if coord[0] < img.shape[1] and coord[1] < img.shape[0] and coord[
                    0] > 0 and coord[1] > 0:
                cv2.circle(tmp_img, tuple(coord), radius + 2,
                           colors[i % num_colors].tolist(), 2)

    return tmp_img

def scalecam(cam, imres):
    if imres == 1:
        return cam
    # Returns a camera which shares camera parameters by reference,
    # and whose scale is resized according to imres
    return ProjectPoints(
        rt=cam.rt,
        t=cam.t,
        f=array(cam.f.r * imres),
        c=array(cam.c.r * imres),
        k=cam.k)

def resize_img(img, scale_factor):
    new_size = (np.round(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    return new_img

