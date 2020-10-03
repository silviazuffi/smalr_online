import numpy as np

def laplacian(part_mesh):
    """ Compute laplacian operator on part_mesh. This can be cached.
        Code from G. Pons Moll
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    from psbody.mesh.topology.connectivity import get_vert_connectivity

    connectivity = get_vert_connectivity(part_mesh)
    # connectivity is a sparse matrix, and np.clip can not applied directly on
    # a sparse matrix.
    connectivity.data = np.clip(connectivity.data, 0, 1)
    lap = normalize(connectivity, norm='l1', axis=1)
    lap = sp.eye(connectivity.shape[0]) - lap
    return lap

