from os.path import join, exists
import pickle as pkl
import numpy as np
import chumpy as ch

class MultiShapePrior(object):

    def __init__(self, family_name='tiger', data_name=''):
        print('Using shape data %s' % data_name)
        data_path = join('../../smpl_models', data_name)
           
        with open(data_path, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
        if family_name == 'tiger' or family_name == 'big_cats':
            family_id = 0
        elif family_name == 'dog':
            family_id = 1
        elif family_name == 'horse':
            family_id = 2
        elif family_name == 'cow':
            family_id = 3
        elif family_name == 'hippo':
            family_id = 4
        else:
            print('Dont know animal %s!' % family_name)
            import ipdb; ipdb.set_trace()

        mean = data['cluster_means'][family_id]
        cov = data['cluster_cov'][family_id]
        invcov = np.linalg.inv(cov + 1e-5 * np.eye(cov.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.cov = cov
        self.mu = mean
        self.prec = prec

    def __call__(self, x):
        # Mahalanobis.
        return (x - self.mu[:len(x)]).dot(self.prec[:len(x), :len(x)])

