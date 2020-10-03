import numpy as np

def setup_camera(w, h, flength=2000, rt=np.zeros(3), t=np.zeros(3), k=np.zeros(5)):
    '''Width is image.shape[0]!!'''
    from opendr.camera import ProjectPoints
    center = np.array([h / 2, w / 2])
    f = np.array([flength, flength])
    cam = ProjectPoints(
        f=f, rt=rt, t=t, k=k, c=center)

    return cam
