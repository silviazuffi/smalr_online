function made = exists_or_mkdir(path)
% function made = exists_or_mkdir(path)
% Make directory path if it does not already exist.

% Obtained from voc-release5

  made = false;
  if exist(path) == 0
    unix(['mkdir -p ' path]);
    made = true;
  end
end
