%set all paths of sub-directories. Using relative path-names, assuming that
%this script is in the root folder 
clc
fprintf(['\nSetting up paths for code-package PopSpikeDyn\n'])

addpath([pwd,'/core/VariationalInferenceDualLDS'])
addpath([pwd,'/core/PLDS'])
addpath([pwd,'/core/LDS'])
addpath([pwd,'/core'])
addpath([pwd,'/utils'])


use_our_minfunc=input(['\nIf you do not have a working version of minFunc by Mark Schmidt installed, \n' ...
    'and want to use the one packaged with this code (version 2012), please type "Y"...\n'],'s');
use_our_minfunc=strcmpi(use_our_minfunc,'y');

if use_our_minfunc
    addpath([pwd,'/utils/minFunc/minFunc'])
    addpath([pwd,'/utils/minFunc/minFunc/compiled'])
    try
        mcholC(eye(3));
        disp('Execution of mex-file mcholC worked, so it seems as if functions are compiled for your system.')
    catch
        disp('Execution of mex-file mcholC failed, seems as if the functions are not compiled for your system.')
        disp('Go to ./utils/minFunc/mex and compile the mex-files therein');
    end
end


