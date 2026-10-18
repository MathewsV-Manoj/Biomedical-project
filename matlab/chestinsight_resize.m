function resized = chestinsight_resize(image, targetSize)
% Toolbox-free bilinear image resize for a scalar or [height width] target.
if isscalar(targetSize), targetSize = [targetSize targetSize]; end
image = single(image);
[height,width] = size(image);
if height == targetSize(1) && width == targetSize(2), resized = image; return; end
[xQuery,yQuery] = meshgrid(linspace(1,width,targetSize(2)),linspace(1,height,targetSize(1)));
resized = interp2(image,xQuery,yQuery,'linear');
resized(~isfinite(resized)) = 0;
end
