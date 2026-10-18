function result = chestinsight_predict(imagePath, finding, modelPath)
% Run MATLAB-only neural-network inference and input-gradient saliency.
if nargin < 2 || isempty(finding), finding = 'Cardiomegaly'; end
root = fileparts(fileparts(mfilename('fullpath')));
if nargin < 3 || isempty(modelPath)
    modelPath = fullfile(root,'models','chestinsight_matlab_model.mat');
end
assert(isfile(modelPath),'MATLAB model not found. Run chestinsight_train first.');
loaded = load(modelPath,'model');
model = loaded.model;
assert(any(strcmp(finding,model.labels)),'Unknown finding.');
[x,inputImage] = chestinsight_preprocess(imagePath,model.imageSize);
[scores,cache] = forward(model.network,x);
index = find(strcmp(finding,model.labels),1);
saliency = saliencyMap(model.network,cache,index,model.imageSize);
result = struct('labels',{model.labels},'scores',scores(:),'thresholds',model.thresholds(:), ...
    'flags',scores(:)>=model.thresholds(:),'finding',finding,'saliency',saliency, ...
    'inputImage',inputImage,'modelName',model.name,'modelPath',modelPath, ...
    'notice','Research prototype. Scores are not calibrated clinical probabilities and are not for diagnosis or treatment.');
end

function [probabilities,cache] = forward(network,x)
cache.z1 = network.W1*x + network.b1;
cache.a1 = max(cache.z1,0);
cache.z2 = network.W2*cache.a1 + network.b2;
cache.a2 = max(cache.z2,0);
cache.z3 = network.W3*cache.a2 + network.b3;
probabilities = sigmoid(cache.z3);
end

function map = saliencyMap(network,cache,index,imageSize)
gradient3 = zeros(size(network.W3,1),1,'single'); gradient3(index) = 1;
gradient2 = network.W3' * gradient3;
gradient2 = gradient2 .* single(cache.z2 > 0);
gradient1 = network.W2' * gradient2;
gradient1 = gradient1 .* single(cache.z1 > 0);
gradientInput = abs(network.W1' * gradient1);
map = reshape(gradientInput,imageSize,imageSize);
map = map - min(map(:));
if max(map(:)) > 0, map = map / max(map(:)); end
end

function value = sigmoid(x)
x = max(min(x,30),-30);
value = 1 ./ (1 + exp(-x));
end
