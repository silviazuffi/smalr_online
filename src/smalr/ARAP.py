'''

    Author: Silvia Zuffi
    Notes: some code from Javier Romero

'''

from chumpy import Ch
import chumpy as ch
import cv2
import scipy.sparse as sp
import numpy as np
import pdb
from chumpy.ch import MatVecMult
import pickle as pkl
from os.path import join

def wedges(A3, v):
    elength = np.sqrt(np.linalg.norm(A3.T.dot(v.ravel()).reshape(-1,3), axis=1))
    return elength

def edgesIdx(nV, f, save_dir, name):
    # We need to have the same number of vertexes (neighbors)
    # in each cell to compute the matrices, nN is the max 
    # number of neighbors we consider for each vertex 

    nN = 6
    Idx = [np.where(f==i)[0] for i in range(nV)]
    nIdx = [np.unique((f[Idx[i],:].ravel())[np.where(f[Idx[i],:].ravel()!= i)[0]])[:nN] for i in range(len(Idx))]
    try:
        A3 = pkl.load(open(join(save_dir, name + "_A3.pkl"), 'rb'), encoding='latin1')
        A = pkl.load(open(join(save_dir, name + "_A.pkl"), 'rb'), encoding='latin1')
        save_A = False
        return nIdx, A3, A
    except:
        print('computing A and A3')
        save_A = True
        pass
    
    # Define an adiacency matrix to compute the edges
    # given the vertexes. Assumes all the vertexes
    # have the same number of neighbors, nN
    nE = nV*nN

    A3 = np.zeros((nV*3, nE*3), dtype=np.int8)
    for i in range(nV):
        #if len(nIdx[i]) < nN:
        #    print len(nIdx[i])
        for j in range(len(nIdx[i])):
            e = i*nN*3
            A3[i*3, e+j] = -1
            A3[i*3+1, e+j+1] = -1
            A3[i*3+2, e+j+2] = -1
            A3[nIdx[i][j]*3, e+j] = 1
            A3[nIdx[i][j]*3+1, e+j+1] = 1
            A3[nIdx[i][j]*3+2, e+j+2] = 1
    A3 = sp.csc_matrix(A3, dtype=np.int8)
    if save_A:
        print('saving A3')
        pkl.dump(A3, open(join(save_dir, name + "_A3.pkl"), 'wb'))


    A = np.zeros((nV, nE), dtype=np.int8)
    for i in range(nV):
        #if len(nIdx[i]) < nN:
        #    print len(nIdx[i])
        for j in range(len(nIdx[i])):
            e = i*nN*3
            A[i, i*nN+j] = -1
            A[nIdx[i][j], i*nN+j] = 1

    A = sp.csc_matrix(A, dtype=np.int8)
    if save_A:
        print('saving A')
        pkl.dump(A, open(join(save_dir, name + "_A.pkl"), 'wb'))
    return nIdx, A3, A

class LoopProd(Ch):
    # Code from Javier Romero
    ''' compute einsum('ik,ikl->il', edge, rot_3d), where rot_3d is rot.reshape(-1, 3, 3)
        rot has shape nverts x 4, edge has shape nverts x 4 '''

    dterms = ['rot', 'edge']

    def on_changed(self, which):
        if 'edge' in which:
            j = np.arange(self.edge.size)
            i = np.tile(np.arange(self.edge.shape[0]), (self.edge.shape[1],1)).T.flatten()
            self.edge_sp = sp.csc_matrix((self.edge.ravel(), (i,j)))

    def compute_r(self):
        # XXX what is faster, converting into sparse or using einsum?
        return(np.asarray(self.edge_sp.dot(self.rot)))

    def compute_dr_wrt(self, wrt):
        if wrt is self.edge:
            if not hasattr(self, 'i_vh') or not hasattr(self, 'j_vh'):
                iblock, jblock = np.meshgrid(range(self.rot.shape[1]), range(self.edge.shape[1]))
                self.i_vh = np.hstack([(iblock + self.rot.shape[1]*ib).flatten() for ib in range(len(self.edge))])
                self.j_vh = np.hstack([(jblock + self.edge.shape[1]*ib).flatten() for ib in range(len(self.edge))])
            data_vh = self.rot.r.flatten()
            data_vh[3::4] = 0. # put to zero the derivative of all homogeneus constant elements
            return(sp.csc_matrix((data_vh, (self.i_vh, self.j_vh))))
        if wrt is self.rot:
            if not hasattr(self,'i_wM') or not hasattr(self,'j_wM'):
                self.i_wM = np.tile(np.arange(self.edge.size).reshape(self.edge.shape), (1,self.rot.shape[1])).flatten()
                self.j_wM = np.arange(self.rot.size)
            data_JwM = np.tile(self.edge.r.flatten(),(self.rot.shape[1],1)).T.flatten()
            return(sp.csc_matrix((data_JwM, (self.i_wM, self.j_wM))))

class ARAP(Ch):
    dterms = 'reg_e', 'model_e'
    terms = 'A', 'w'

    def on_changed(self, which):
        if 'w' in which or 'A' in which:
            A = np.abs(self.A.tocoo())
            self.wmat = A.tocsc().dot(sp.diags([self.w], [0]))
            A9_data = np.tile(A.data/2., 9)
            A9_row = np.hstack([A.row*9 + i for i in range(9)])
            A9_col = np.hstack([A.col*9 + i for i in range(9)])
            self.A9 = sp.csc_matrix((A9_data, (A9_row, A9_col)),
                                    shape=(A.shape[0]*9, A.shape[1]*9))

            wmatcoo = self.wmat.tocoo()
            wmat3_data = np.tile(wmatcoo.data, 3)
            wmat3_row = np.hstack([wmatcoo.row*3 + i for i in range(3)])
            wmat3_col = np.hstack([wmatcoo.col*3 + i for i in range(3)])
            self.wmat3 = sp.csc_matrix((wmat3_data, (wmat3_row, wmat3_col)),
                                       shape=(wmatcoo.shape[0]*3,
                                              wmatcoo.shape[1]*3))

        if 'reg_e' in which or 'model_e' in which:
            T = np.einsum('ij,jk->ikj',
                          self.model_e.T, self.reg_e).reshape(9, -1)

        # compute rotation per vertex rv through SVD
        T_per_v = (self.wmat.dot(T.T)).reshape(-1, 3, 3)
        u_s_v = np.linalg.svd(T_per_v)
        col3 = np.asarray([1., 1., -1.])
        # negate third column of u if det is negative
        # The real rotation, according to Sorkine, would be v.T.dot(u.T),
        # because numpy returns usv instead of usv.T. However, we're applying
        # e.dot(rot), so we need rot.T which is u.dot(v)
        self.rv = np.asarray([u.dot(v) if np.linalg.det(u.dot(v)) > 0 # np.prod(s) > 0#
                              else (u*col3).dot(v) for u, s, v in zip(*u_s_v)])

        # average vertex rotations for each edge
        self.rf = MatVecMult(mtx=self.A9.T, vec=self.rv.ravel()).reshape(-1, 3)

        w3 = ch.tile(self.w, (3, 1)).T.ravel()
        self.err = (LoopProd(rot=self.rf, edge=self.model_e).ravel()
                    - self.reg_e.ravel())*w3

    def compute_r(self):
        return(self.err.r)

    def compute_dr_wrt(self, wrt):
        return(self.err.dr_wrt(wrt))


