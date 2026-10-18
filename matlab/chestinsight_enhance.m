function enhanced = chestinsight_enhance(image)
% Toolbox-free 1st-to-99th percentile contrast stretch for display only.
image = chestinsight_gray(image);
values = sort(image(:)); count = numel(values);
low = values(max(1,round(0.01*count))); high = values(min(count,max(1,round(0.99*count))));
enhanced = max(0,min(1,(image-low)/max(high-low,eps('single'))));
enhanced = uint8(round(255*enhanced));
end
