function res = get_yes()

[~, ~, button_double] = ginput(1);
res = button_double == 'y';
if button_double == 'b'
    fprintf('keyboard in get_yes()\n');
    keyboard;
end
