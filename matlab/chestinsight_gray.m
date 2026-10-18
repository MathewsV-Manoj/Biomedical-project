function image = chestinsight_gray(image)
% Toolbox-free grayscale conversion and normalization to single [0,1].
if ndims(image) == 3
    image = 0.2989*single(image(:,:,1)) + 0.5870*single(image(:,:,2)) + 0.1140*single(image(:,:,3));
else
    image = single(image);
end
if isinteger(image)
    image = image / single(intmax(class(image)));
elseif max(image(:)) > 1
    image = image / max(image(:));
end
minimum = min(image(:)); maximum = max(image(:));
if maximum > minimum
    image = (image-minimum)/(maximum-minimum);
else
    image = zeros(size(image),'single');
end
end
