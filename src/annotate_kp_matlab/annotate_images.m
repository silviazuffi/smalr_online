%
%   Author: Angjoo Kanazawa
%
%
%

function annotate_images()

base_dir = './images'

% family = 'dog';
%family = 'hippo';
% family = 'cow';
%family = 'others';
family = 'big_cats';
%family = 'horse';
% base_dir = fullfile(base_dir, family);

ext = '(jpg|JPG|png)';

content = dir(base_dir);
names = {content.name};
ok = regexpi(names, ['.*\.',ext,'$'],'start');
names = names(~cellfun(@isempty,ok));
% Remove silhouettes.
ok = regexpi(names, ['_s.png$'],'start');

do_these = names(cellfun(@isempty,ok));

load('ferrari2smpl_tiger_wtail_wears_new_tailstart_wnose_whtail.mat', 'map')

anno_name = '_ferrari-tail-nose-htail';

smpl2name = map;

% Model:
model_path = './plys/tiger_smpl.ply';
if strcmp(family, 'dog')
    model_path = './plys/dog_smpl.ply';
elseif strcmp(family, 'cow')
    model_path = './plys/cow_smpl.ply';
elseif strcmp(family, 'hippo')
    model_path = ['./plys/' ...
                  'hippo_smpl.ply'];
elseif strcmp(family, 'horse') || strcmp(family, 'others')
    model_path = ['./plys/' ...
                  'horse_smpl.ply'];
end


for i = 1:length(do_these)
    img_path = fullfile(base_dir, do_these{i});
    [par, fname, ~] = fileparts(img_path);
    out_dir = fullfile(par, 'annotations');
    exists_or_mkdir(out_dir);
    
    out_path = fullfile(out_dir, [fname anno_name '.mat']);
    fprintf('%d/%d: image %s\nout %s\n', i, length(do_these), img_path, out_path);
    annotate2DwithQuery_multipleverts(img_path, model_path, smpl2name, ...
                        out_path);
end
