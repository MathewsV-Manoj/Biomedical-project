function [feature, displayImage] = chestinsight_preprocess(inputImage, imageSize)
% Convert an X-ray to the exact MATLAB-only model input.
if nargin < 2, imageSize = 32; end
if ischar(inputImage) || isstring(inputImage)
    inputImage = imread(inputImage);
end
inputImage = chestinsight_gray(inputImage);
resized = chestinsight_resize(inputImage,[imageSize imageSize]);
feature = resized(:);
displayImage = resized;
end
