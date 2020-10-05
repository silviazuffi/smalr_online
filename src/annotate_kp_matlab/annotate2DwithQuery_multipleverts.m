function annotate2DwithQuery_multipleverts(img_path, model_path, map, out_path)
% Asks users where the pre-defined vertex on the 3D model is.
% map struct that contains 'names': part names & 'vids': cell of vertex ids.
% Multiple verts are allowed in this one. 
% Saves just the kp, which stores num_parts x 2, invisible: num_parts x 1
% and names : num_parts x 1.
% Doesn not save 'cam_info' or 'v_ids' 

if ~exist('read_ply', 'file');
    addpath('libs');
end
num_parts = length(map.names);
if exist(out_path, 'file')
    t = load(out_path);
    if isfield(t, 'landmarks')
        % This is ferrari's notation, break it up., add one for tail
        kp = t.landmarks(:, 1:2);
        invisible = ~t.landmarks(:, 3);
        if num_parts > size(kp, 1)
            kp = [kp; 0 0];
            invisible = [invisible; -1];
        end
        annotation = struct('kp', kp, 'invisible', invisible, 'names', ...
                            {map.names}, 'on_ground', -ones(num_parts, 1));
        % Resave.
        save(out_path, 'annotation');
    else
        annotation = t.annotation;
    end
else
    % -1 means it's not labeled.
    annotation = struct('kp',zeros(num_parts, 2),...
                        'invisible', -ones(num_parts, 1), 'names', ...
                        {map.names});
end



map.colors = distinguishable_colors(num_parts, {'k', 'w'});

names = map.names;

mesh = get_mesh(model_path, map);
I = imread(img_path);
if ndims(I) == 2
    I = imadjust(I);
    I = repmat(I, [1, 1, 3]);
end
[nRow, nCol, ~] = size(I);


if size(annotation.kp, 1) < num_parts
    annotation.kp = [annotation.kp; 0 0];
    [h_fig, h_title] = draw_2D(I, annotation, map.colors);
    set(h_title, 'string', 'Noo its off by some.. check the colors are right?')
    h_fig3D = draw_3D(mesh, annotation, []);
    keyboard
    save(out_path, 'annotation')
end

if size(annotation.invisible,1) == 1
    invisible = annotation.invisible';
else
    invisible = annotation.invisible;
end
if any(all(annotation.kp== 0,2) & invisible == 0)
    % If any are visible, but have [0, 0] remove them.
    bad_inds = find(all(annotation.kp== 0,2) & invisible == 0);
    annotation.invisible(bad_inds) = -1;
    save(out_path, 'annotation');
end


redraw = true;
done = false;
h_fig3D = [];
while ~done
    if redraw
        if ~isempty(h_fig3D)
            % Update 3D image.
            sfigure(h_fig3D);
            mesh.cam_pos = campos();
            mesh.cam_tgt = camtarget();
        end
        [h_fig, h_title] = draw_2D(I, annotation, map.colors);
        h_fig3D = draw_3D(mesh, annotation, []);
    end
    %% Get list of not clicked v_ids:
    not_labeled = annotation.invisible == -1;
    if all(not_labeled == 0) 
        fprintf('Done!!\n');
        done = true;
        continue;
    end
    if any(not_labeled == 1)
        vid2ask = find(not_labeled, 1);
        name2ask = strrep(names{vid2ask}, '_', ' ');
        % draw_3D(mesh, annotation, vid2ask, name2ask);
    else
        vid2ask = [];
        name2ask = '--';
        % fprintf('Annotate camera viewpoint!')
        fprintf('Anything to fix?')
    end

    sfigure(1);
    % Draw this point in 3D
    set(h_title, 'string', sprintf(['Where is "%s" (the start points) in 3D? type "n" if invisible, ' ...
                        '"r" to remove already clicked,\n"c" to annotate viewpoint "q" to move on ' ...
                        'to the next image, "k" to break.'], name2ask));
    set(0, 'currentfigure', h_fig);
    
    
    [col_in, row_in, button_in] = ginput(1);            
    col_in = round(col_in);
    row_in = round(row_in);
    if isempty(vid2ask) && (isempty(strfind('rckzq', button_in)))
        fprintf('input either Remove, Continue, z, k, or q\n');
        continue
    end
    % If in bounds draw.
    if (button_in == 'n')
        fprintf('%s (%d) invisible!\n', names{vid2ask}, vid2ask)
        annotation.invisible(vid2ask) = 1;
        save(out_path, 'annotation');
    elseif (button_in == 'r')    
        % Remove mode.
        fprintf('Remove mode!, type any character if its for invisible point\n');
        set(h_title, 'string', 'Click a point in figure to REMOVE');
        [col_in, row_in, button_in] = ginput(1);
        if (button_in == 1)
            col_in = round(col_in);
            row_in = round(row_in);
            % If in bounds draw.
            if ((col_in > 0) && (row_in > 0) && ...
                (col_in <= nCol && (row_in <= nRow)))
                %% Find closest.
                [~, anno_id] = min(sum(bsxfun(@minus, annotation.kp,...
                                              [col_in row_in]).^2,2));
                selected_pt = annotation.kp(anno_id,:);
                plot(selected_pt(1), selected_pt(2), 'o', 'MarkerSize', 5, 'MarkerEdgeColor', ...
                     'y', 'MarkerFaceColor', 'r');
                set(h_title, 'string', 'Remove this location? (y/n)');
                if (get_yes())
                    annotation.kp(anno_id,:) = [0 0];
                    % TODO: CHANGE VISIBILITY
                    annotation.invisible(anno_id) = -1;
                    save(out_path, 'annotation');
                end
            end
        else
            set(h_title, 'string', 'Redo an invisible point?')
            fprintf('Redo an invisible point?')
            if (get_yes())
                invisible_points = find(annotation.invisible == 1);
                for j = 1:length(invisible_points)
                    fprintf('%s: %d\t', names{invisible_points(j)}, invisible_points(j));
                end
                fprintf('\n');
                x = input('Pick from above');
                if any(x == invisible_points)
                    fprintf('Resetting %s (%d) visibility\n', names{x}, x);
                    annotation.invisible(x) = -1;
                    save(out_path, 'annotation');
                else
                    disp x;
                    fprintf('input is not one of the choices!\n');
                    keyboard
                end
                
            end
        end
        fprintf('End remove mode.\n');
    elseif (button_in == 'c')    
        fprintf('Camera mode!\n');
        % Go to 3D show current camera mode.
        set(h_title, 'string', ['Go to 3D and view from ' ...
                            'camera']);
        cam_info = get_3D_camera(mesh, annotation);
        h_fig = sfigure(11); clf;
        patch('Faces',mesh.tri,'Vertices',mesh.X,'EdgeColor',[0.7 0.7 0.7], ...
              'FaceColor','w','Marker', '.', 'MarkerSize', ...
              5, 'MarkerEdgeColor', 'k');
        hold on;
        axis image off;
        camtarget(cam_info.camtarget);
        campos(cam_info.campos);
        camup(cam_info.camup);
        title('new cam setting, Is this camera good?(y/n)');
        if (get_yes())
            annotation.cam_info = cam_info;
            save(out_path, 'annotation');
        end
        fprintf('End camera mode!\n');
    elseif (button_in == 'z')    
        fprintf('pausing\n');
        % Zoom while keyboarding, type something in matlab to continue.
        pause;
        redraw = false;
    elseif (button_in == 'k') 
        keyboard;
        redraw = false;
    elseif (button_in == 'q') 
        done = true;
    elseif ((col_in > 0) && (row_in > 0) && ...
            (col_in <= nCol && (row_in <= nRow)))
        plot(col_in, row_in, 'o', 'MarkerSize', 5, 'MarkerEdgeColor', ... $ Silvia was 5
             'w', 'MarkerFaceColor', 'g');
        % set(h_title, 'string', 'Accept this location? (y/n)');
        if true %(get_yes())
            % saving as (x, y)
            new_point = [col_in row_in];
            annotation.kp(vid2ask, :) = new_point;
            annotation.invisible(vid2ask) = 0;
            save(out_path, 'annotation');
        end
    else
        fprintf('Point clicked out of bounds, redo\n');
    end
    redraw = true;
end


function mesh = get_mesh(fname, map)
[~, ~, ext] = fileparts(fname);

if strcmp(ext, '.ply')
    [X, tri] = read_ply(fname);
else
    [X, tri] = read_off(fname);
    X = X';
    tri = tri';
end

mesh.X = X;
mesh.tri = tri;
mesh.cam_tgt = [0 0 0];
mesh.cam_pos = [2.4054    7.5755    1.8227];
mesh.annotation.vids = map.vids;
mesh.annotation.colors = map.colors;

function h_fig = draw_3D(mesh, annotation, show_vid, show_name)
% If show_vid is supplied, highlight that vertex.  
% Mesh contains the list of pre-defined vertices. 
if nargin < 3
    show_vid = [];
end
h_fig = sfigure(2); clf;

patch('Faces',mesh.tri,'Vertices',mesh.X,'EdgeColor',[0.7 0.7 0.7], ...
      'FaceColor','w','Marker', '.', 'MarkerSize', ...
      5, 'MarkerEdgeColor', 'k');
hold on;
% Plot possible mesh points
vids = mesh.annotation.vids;
if ~isempty(vids)
    colors = mesh.annotation.colors;
    vis = annotation.invisible == 0;
    for i = 1:length(vids)
        show = mesh.X(vids{i}, :);
        % Plot already correspondende points with diamond
        if vis(i) == 1 
            scatter3(show(:, 1), show(:, 2), show(:, 3), 100, colors(i,:), 'd',...
                     'filled')
        else
            scatter3(show(:, 1), show(:, 2), show(:, 3), 100, colors(i,:), ...
                     'filled')
        end
        % plot3(mesh.annotation.threeD(ok, 1), mesh.annotation.threeD(ok, 2), mesh.annotation.threeD(ok, 3), 'o', 'MarkerSize', 10, 'MarkerEdgeColor', ...
        %       'w', 'MarkerFaceColor', 'r');
    end
end

if ~isempty(show_vid)
    show = mesh.X(vids{show_vid}, :);
    scatter3(show(:, 1), show(:, 2), show(:, 3), 500, colors(show_vid,:), 'h',...
             'filled')
    % plot3(show(:, 1), show(:, 2), show(:, 3), 'o', 'MarkerSize', 9, 'MarkerEdgeColor', ...
    %       'k', 'MarkerFaceColor', 'b');    
    title((show_name))
end

set(gcf, 'color', 'w')
axis image off;
cameratoolbar;
camtarget(mesh.cam_tgt);
campos(mesh.cam_pos);


function [h_fig, h_title] = draw_2D(I, annotation, colors)
h_fig = sfigure(1); clf;
curr_pos = get(h_fig, 'Position');
if curr_pos(3) < 1400
    set(h_fig, 'Position', [curr_pos(1:2) 1400 900]);
end
imagesc(I); hold on;
vis = annotation.invisible == 0;
if ~isempty(vis)
    scatter(annotation.kp(vis, 1), annotation.kp(vis, 2), 50, colors(vis, :),  'filled', 'LineWidth', 1);

    set(h_fig, 'name', sprintf('%d points annotated', sum(vis)));
end
axis image off;
set(gcf, 'color', 'w')
h_title = title('');

function [part_id, new_point, mesh, v_id] = get_3D_index(mesh, annotation)
% Draw
h_fig = draw_3D(mesh, annotation);
set(0, 'currentfigure', h_fig);
dcmObj = datacursormode(h_fig);
set(dcmObj, 'DisplayStyle', 'datatip', 'SnapToDataVertex', 'on', 'Enable', 'on');
pause; % when done selecting, type anything in matlab to
       % get back.
point1 = getCursorInfo(dcmObj);        

if isempty(point1)
    part_id = -1;
    v_id = -1;
    new_point = [];
    return;
end
x = point1.Position(1);
y = point1.Position(2);
z = point1.Position(3);

%% Assign to closest part:
% TODO CHECK:
[~, part_id] = min(sum(bsxfun(@minus, mesh.annotation.threeD, [x y ...
                    z]).^2,2));
selected_pt = mesh.annotation.threeD(part_id,:);
v_id = mesh.annotation.v_ids(part_id);
% fprintf('click @ (%f, %f, %f)\n', x,y,z);        
sfigure(h_fig);
plot3(selected_pt(1), selected_pt(2), selected_pt(3), 'o', 'MarkerSize', 15, 'MarkerEdgeColor', ...
      'w', 'MarkerFaceColor', 'g');
% update current camera position.
mesh.cam_pos = campos();
mesh.cam_tgt = camtarget();

new_point = selected_pt;


function cam_info = get_3D_camera(mesh, annotation)

if isfield(annotation,'cam_info') && ~isempty(annotation.cam_info);
    h_fig = sfigure(10); clf;
    patch('Faces',mesh.tri,'Vertices',mesh.X,'EdgeColor',[0.7 0.7 0.7], ...
          'FaceColor','w','Marker', '.', 'MarkerSize', ...
          5, 'MarkerEdgeColor', 'k');
    hold on;
    axis image off;
    camtarget(annotation.cam_info.camtarget);
    campos(annotation.cam_info.campos);
    camup(annotation.cam_info.camup);
    title('current cam setup');
end

h_fig = draw_3D(mesh, annotation);
set(0, 'currentfigure', h_fig);
pause; % when done selecting, type anything in matlab to

cam_info.campos = campos();
cam_info.camup = camup();
cam_info.camtarget = camtarget();
c = cam_info.campos./norm(cam_info.campos);
b = cam_info.camup./norm(cam_info.camup);
cam_info.rotation = [cross(b, c); b; c];
% This is the rotation to be set in SMPL pose[:3].
cam_info.R = [cross(-b, -c); -b; -c]

camtarget(mesh.cam_tgt);
campos(mesh.cam_pos);
