code = 'online' 
clean_from_green = False 

if code == 'online':
    smalr_dir='smalr_output'
    model_name = 'smal_00781_4_all.pkl'
    shape_data_name = 'smal_data_00781_4_all.pkl'
    animal_output_dir = 'animal_output'
    settings = {
    'mean_pose_prior':'../../pose_priors/walking_toy_symmetric_35parts_mean_pose.npz',
    'pose_prior': 'walking_toy_symmetric_pose_prior_with_cov_35parts.pkl',
    'tail_pose_prior': 'walking_toy_symmetric_pose_prior_with_cov_35parts.pkl',
    'k_shape_term': 1e4, #0.8*1e3,
    'k_betas_var': 1e5,
    'k_pose_term': 2*1e4, #4*1e3,
    'k_tail_pose_term': 1e3,
    'k_rest_pose_term': 4*1e3,
    'k_m2s':10 * 1e3,
    'k_s2m':2 * 1e3, 
    'k_kp_term': 1.5*1e3,
    'k_limit_term' : 7 * 1e5,
    'k_trans_term' : 1e3,
    'k_rot_term' : 1e5,
    'k_robust_sig': 150,
    'k_pose_term_free_shape': 2 * 1e4,
    'k_shape_term_free_shape' : 2*1e5,
    # Refinement
    'ref_k_arap_per_view': 1.5*1e2,
    'ref_W_arap_parts':['Head', 'Mouth', 'LEar', 'REar'],
    'ref_W_arap_values':[0.5, 0.5, 0.5, 0.5],
    'ref_k_lap':1.5*2*1e1,
    'ref_k_sym':2*1e2,
    'ref_k_keyp_weight':2*1e-4,
    'ref_k_kp_term': 2.2*1e2, #1.5*1e3,
    'ref_k_shape_term': 1e3,
    'ref_k_pose_term': 1e4,
    'ref_k_rest_pose_term': 1.5 * 1e3,
    'ref_k_limit_term': 7 * 1e5,
    'ref_betas_var': 1e5,
    'ref_k_pose_term_free_shape': 2.2*1e2, #2*1e4,
    'ref_k_shape_term_free_shape': 2*1e5,
    'ref_k_trans_term': 1e3,
    'ref_k_rot_term': 1e5,
    'ref_k_silh_term': 1,
    'ref_k_m2s':8, #10,
    'ref_k_s2m':2, #4,
    'max_tex_weight': True, # use True for zebra and cheetah
    'output_replace_mouth_with_prediction':False,
    'template_w_tex_uv_name':'smal_00781_4_all_template_w_tex_uv_001.obj',
    'texture_map_colored_name':'smal_00781_4_all_template_w_tex_uv_001_colored.png',
    'texture_color_locations':([316,62,204,326,290,338,390,448,450,488,548,534,764,798,852,884,591,590,719],[536,216,184,158,70,64,168,190,40,50,78,50,204,196,270,506,105,65,586]),
    }

