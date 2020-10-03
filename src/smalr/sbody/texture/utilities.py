from psbody.mesh import Mesh
from psbody.mesh.geometry.barycentric_coordinates_of_projection import barycentric_coordinates_of_projection
from psbody.mesh.topology.connectivity import get_vert_connectivity
import numpy as np
import scipy
import cv2
import glob
import os

def uv_to_xyz_and_normals(alignment, face_indices_map, b_coords_map):
	if not hasattr(alignment, 'vn'):
		alignment.reset_normals() 

	pixels_to_set = np.array(np.where(face_indices_map != -1)).T
	x_to_set = pixels_to_set[:,0]
	y_to_set = pixels_to_set[:,1]
	b_coords = b_coords_map[x_to_set,y_to_set,:]
	f_coords = face_indices_map[x_to_set, y_to_set].astype(np.int32)
	v_ids = alignment.f[f_coords]
	points = np.tile(b_coords[:,0],(3,1)).T*alignment.v[v_ids[:,0]] +\
					 np.tile(b_coords[:,1],(3,1)).T*alignment.v[v_ids[:,1]] +\
					 np.tile(b_coords[:,2],(3,1)).T*alignment.v[v_ids[:,2]]
	normals = np.tile(b_coords[:,0],(3,1)).T*alignment.vn[v_ids[:,0]] +\
					 np.tile(b_coords[:,1],(3,1)).T*alignment.vn[v_ids[:,1]] +\
					 np.tile(b_coords[:,2],(3,1)).T*alignment.vn[v_ids[:,2]]				 
	return (points, normals)				 


def generate_template_map_by_triangles(template, map_scale=1.):
	map_height = map_width = int(2048*map_scale)

	face_indices_map = np.ones((map_height, map_width, 3)) * -1
	text_coords = template.vt[template.ft.flatten()][:,:2]    
	text_coords *= np.tile([map_width, -map_height],(template.ft.size,1))
	text_coords += np.hstack((np.zeros((template.ft.size,1)),map_height*np.ones((template.ft.size,1))))
	text_coords_tr = text_coords.reshape(-1,6).astype(np.int32)
	# XXX fillConvexPoly seems like an overkill for drawing (many) triangles
	# We should either draw them in opengl or convert it to c++
	for itc, tc in enumerate(text_coords_tr):
		cv2.fillConvexPoly(face_indices_map, tc.reshape(3,2), [itc,itc,itc])

	face_indices_map = face_indices_map[:,:,0]

	pixels_to_set = np.array(np.where(face_indices_map != -1)).T

	x_to_set = pixels_to_set[:,0]
	y_to_set = pixels_to_set[:,1]
	f_indices = face_indices_map[x_to_set,y_to_set].astype(np.int32)

	# new method
	text_coords = np.reshape(np.fliplr(text_coords), (-1,6)).astype(np.int32)[f_indices]

	points = np.hstack( (pixels_to_set, np.zeros( (pixels_to_set.shape[0],1) )))
	
	first_v = np.hstack((text_coords[:,0:2], np.zeros((text_coords.shape[0],1))))
	second_v = np.hstack((text_coords[:,2:4], np.zeros((text_coords.shape[0],1))))
	third_v = np.hstack((text_coords[:,4:6], np.zeros((text_coords.shape[0],1))))

	b_coords = barycentric_coordinates_of_projection(points, first_v, second_v-first_v, third_v-first_v)
	b_coords_flat = b_coords.flatten()    
	wrong_pixels = np.union1d(np.where(b_coords_flat<0)[0], np.where(b_coords_flat>1)[0])
	wrong_ids = np.unique(np.floor(wrong_pixels/3.0).astype(np.int32))

	b_coords_map = np.ones((map_height, map_width, 3)) * -1
	b_coords_map[x_to_set,y_to_set,:] = b_coords
	
	return (face_indices_map, b_coords_map)    

