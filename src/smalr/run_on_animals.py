from os.path import join 
from compute_clips import compute_clips
from psbody.mesh import Mesh
from smalr_settings import model_name, shape_data_name, smalr_dir, animal_output_dir

import numpy as np
np.random.seed(0)

save_base_dir = '../../'+smalr_dir+'/'+animal_output_dir+'/'
base_dir = '../../data/'

# Read the symmetric indexes from the gloss model. They are needed to set the texture.
import pickle as pkl
symIdx = pkl.load(open('symIdx.pkl','rb'), encoding='latin1')

max_image_size = 480
animals_set = ['cheetah_D'] #'zebra_A'] 

init_flength = 1000.
align_w_optimized_template = False

landmarks = None

for animal in animals_set:
    if animal == 'tasmanian_tiger_A':
        border = 50
        use_face_landmarks = True
        no_tail = False
        family = 'other'
        use_face_landmarks = True
        base_dir = '../../data/'
        clip_set = ['01', '01', '01', '01'] 
        fStart = ['7020', '7181', '7005', '7584']
        fStop = ['7020', '7181', '7005', '7584']
        clip_set = ['01', '01', '01', '01', '01', '01']
        fStart = ['7020', '7181', '7005', '7552', '7223', '7584']
        fStop = ['7020', '7181', '7005', '7552', '7223', '7584']

    elif animal == 'zebra_A':
        border = 0
        use_face_landmarks = True
        no_tail = False
        family = 'horse'
        base_dir = '../../data/'
        clip_set = ['01','01', '01', '01', '01']
        fStart = ['0000', '0001', '0002', '0003', '0004']
        fStop = ['0000', '0001', '0002', '0003', '0004']
        clip_set = ['01','01', '01', '01']
        fStart = ['0000', '0001', '0002', '0003']
        fStop = ['0000', '0001', '0002', '0003']

    elif animal == 'cheetah_C' or animal == 'cheetah_D':
        border = 100
        no_tail = False
        landmarks = 'nose_htail'
        use_face_landmarks = False
        base_dir = '../../data/'
        family = 'big_cats'
        clip_set = ['01']
        fStart = ['0000']
        fStop = ['0000']


    animal_set = [animal]*len(clip_set)
    base_dirs = [join(base_dir, ase) for ase in animal_set]

    frameStarts = [join(base_dirs[i], animal_set[i]+'_'+clip_set[i]+'/frame'+fs+'.png') for i,fs in enumerate(fStart)]
    frameStops = [join(base_dirs[i], animal_set[i]+'_'+clip_set[i]+'/frame'+fs+'.png') for i,fs in enumerate(fStop)]

    code = animal_set[0]+'_n_'+str(len(clip_set))+'_'+('_'.join(clip_set))+'_'+('_'.join(fStart))
    if use_face_landmarks:
        code = code + '_face'
    opt_model_dir = join(save_base_dir, code)

    # Align using the previous computed model
    if align_w_optimized_template:
        opti_mesh = Mesh(filename=join(opt_model_dir, 'mesh_v_opt_0.ply'))
        custom_template = opti_mesh.v
    else:
        custom_template = None

    if use_face_landmarks:
        landmarks = 'face'

    compute_clips(family, model_name, shape_data_name, base_dirs, save_base_dir,
             animal_set, clip_set, frameStarts, frameStops, symIdx,
             viz=True, init_flength=init_flength, init_from_mean_pose=True, border=border,
             opt_model_dir=opt_model_dir,
             custom_template=custom_template, NO_TAIL=no_tail, code=code,
             landmarks=landmarks, 
             max_image_size=max_image_size) 

