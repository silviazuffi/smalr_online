from psbody.mesh.meshviewer import MeshViewer, MeshViewers
from psbody.mesh import Mesh
import numpy as np
import cv2
import pickle as pkl
from smalr_settings import model_name, shape_data_name, smalr_dir, animal_output_dir
from os.path import join


animals = ['zebra_A']

for animal in animals:
    if animal == 'tasmanian_tiger_A':
        #location = 'tasmanian_tiger_A_n_4_01_01_01_01_7020_7181_7005_7584_face'
        #frames = ['7020', '7181', '7005', '7584']
        #texture_type = 'filled'
        #pose_location = join('../..', smalr_dir, animal_output_dir, 'tasmanian_tiger_A_01/tracking/'+location)
        #location = join('../..', smalr_dir, animal_output_dir, location)
        location = 'tasmanian_tiger_A_n_6_01_01_01_01_01_01_7020_7181_7005_7552_7223_7584_face'
        frames = ['7020', '7181', '7005', '7552', '7223','7584']
        texture_type = 'filled'
        pose_location = join('../..', smalr_dir, animal_output_dir, 'tasmanian_tiger_A_01/tracking/'+location)
        location = join('../..', smalr_dir, animal_output_dir, location)
    elif animal == 'zebra_A':
        location = 'zebra_A_n_4_01_01_01_01_0000_0001_0002_0003_face'
        frames = ['0000', '0001', '0002', '0003']
        texture_type = 'filled'
        pose_location = join('../..', smalr_dir, animal_output_dir, 'zebra_A_01/tracking/'+location)
        location = join('../..', smalr_dir, animal_output_dir, location)
    elif animal == 'cheetah_C':
        location = 'cheetah_C_n_1_01_0000'
        frames = ['0000']
        texture_type = 'filled'
        pose_location = join('../..', smalr_dir, animal_output_dir, 'cheetah_C_01/tracking/'+location)
        location = join('../..', smalr_dir, animal_output_dir, location)
    elif animal == 'cheetah_D':
        location = 'cheetah_D_n_1_01_0000'
        frames = ['0000']
        texture_type = 'filled'
        pose_location = join('../..', smalr_dir, animal_output_dir, 'cheetah_D_01/tracking/'+location)
        location = join('../..', smalr_dir, animal_output_dir, location)

    mesh_filename_opt = location+'/mesh_v_opt_no_mc_0.ply'
    texture_filename = location+'/texture_final_filled_0_non_opt.png'
    texture_filename_avg = location+'/texture_final_average_0_non_opt.png'
    if texture_type == 'avg':
        texture_filename = texture_filename_avg

    mesh = Mesh(filename=mesh_filename_opt)

    from mycore.io import load_animal_model
    model = load_animal_model(model_name)
    model.v_template[:] = mesh.v
    uv_mesh = Mesh(filename='smal_00781_4_all_template_w_tex_uv_001.obj')

    # show texture
    mesh.ft = uv_mesh.ft
    mesh.vt = uv_mesh.vt
    mesh.set_vertex_colors("white")

    mesh.texture_filepath = texture_filename

    # Read poses
    for frame in frames:
        data = pkl.load(open(join(pose_location, 'frame'+frame+'.pkl'), 'rb'), encoding='latin1')
        pose = data['pose']

        model.pose[:] = pose
        mesh.v = model.r.copy()
        v = mesh.v.copy()
        mesh.v[:,1] = -v[:,1]
        mesh.v[:,2] = -v[:,2]

        mv = MeshViewers(shape=(1,1))
        mv[0][0].set_background_color(np.ones(3))
        mv[0][0].set_static_meshes([mesh])
        import pdb; pdb.set_trace()


