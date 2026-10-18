function data = chestinsight_load_data(forceRebuild)
% Load cached 32x32 MATLAB features made from local NIH pilot images.
if nargin < 1, forceRebuild = false; end
root = fileparts(fileparts(mfilename('fullpath')));
cacheFile = fullfile(root,'data','matlab_features.mat');
labels = chestinsight_labels();
if isfile(cacheFile) && ~forceRebuild
    data = load(cacheFile);
    if isequal(data.labels,labels), return; end
end
manifest = readtable(fullfile(root,'data','manifest.csv'),'TextType','string');
count = height(manifest);
features = zeros(32*32,count,'single');
targets = zeros(numel(labels),count,'single');
for index = 1:count
    [features(:,index),~] = chestinsight_preprocess(fullfile(root,char(manifest.image_path(index))),32);
    targets(:,index) = single(manifest{index,labels})';
    if mod(index,500)==0 || index==count
        fprintf('Preparing MATLAB features: %d/%d\n',index,count);
    end
end
splits = string(manifest.split)';
imagePaths = string(manifest.image_path)';
patientIds = manifest.patient_id';
save(cacheFile,'features','targets','splits','imagePaths','patientIds','labels','-v7.3');
data = load(cacheFile);
end
