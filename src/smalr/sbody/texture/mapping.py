from psbody.mesh import Mesh
import numpy as np
import scipy
import cv2
import sys
import math
from psbody.mesh.visibility import visibility_compute
from sbody.texture.utilities import * 
from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints

def camera_projection(alignment, camera, cam_vis_ndot, image, face_indices_map, b_coords_map, dist=None, masked=False):
	if not hasattr(alignment, 'points'):
		raise AttributeError('Mesh does not have uv_to_xyz points...')

	vis = cam_vis_ndot[0][-alignment.points.shape[0]:]  
	n_dot = cam_vis_ndot[1][-alignment.points.shape[0]:]  
	vis[n_dot<0] = 0
	n_dot[n_dot<0] = 0

	cmap = np.zeros((face_indices_map.shape[0], face_indices_map.shape[1], 3)) if not masked else \
		 np.ones((face_indices_map.shape[0], face_indices_map.shape[1], 3))*-1
	vmap = np.zeros(face_indices_map.shape)

	if len(alignment.points[vis==1]):
		# set camera radial distortion parameters equal to 0, since we are already working on undistorted images		
		#(tmp_proj,J) = cv2.projectPoints(alignment.points[vis==1], camera.r, camera.t, camera.camera_matrix, distCoeffs=np.zeros(5))		
		tmp_proj = camera.r[vis==1]
		im_coords = np.fliplr(np.atleast_2d(np.around(tmp_proj.squeeze()))).astype(np.int32)

		"""if (np.max(im_coords[:,0]) >= np.shape(image)[0]) or (np.max(im_coords[:,1]) >= np.shape(image)[1]):
			print 'error'
			import pdb; pdb.set_trace()"""

		part_face_indices_map = np.copy(face_indices_map)
		not_vis_pixels = np.zeros(len(alignment.points))
		not_vis_pixels[vis==0] = -1
		part_face_indices_map[face_indices_map != -1] = not_vis_pixels
		pixels_to_set = np.array(np.where(part_face_indices_map != -1)).T

		# this check might be unnecessary
		inside_image = np.where(np.logical_and(im_coords[:,0] < np.shape(image)[0], im_coords[:,0]>=0 ) )[0]
		inside_image = np.intersect1d(inside_image, np.where(np.logical_and(im_coords[:,1] < np.shape(image)[1], im_coords[:,1] >= 0))[0])
		pixels_to_set = pixels_to_set[inside_image] 
		im_coords = im_coords[inside_image]
		# end check

		cmap[pixels_to_set[:,0],pixels_to_set[:,1],:] = image[im_coords[:,0],im_coords[:,1],:]
		vmap[pixels_to_set[:,0],pixels_to_set[:,1]] = n_dot[vis==1][inside_image]

	return (cmap, vmap) #, dmap)

