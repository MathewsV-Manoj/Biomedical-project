function overlay = chestinsight_make_overlay(image,saliency)
% Toolbox-free saliency overlay using the built-in parula color map.
base = chestinsight_resize(chestinsight_gray(image),[512 512]);
saliency = chestinsight_resize(saliency,[512 512]);
saliency = max(0,min(1,saliency)); colors = parula(256);
indices = max(1,min(256,round(saliency*255)+1)); heat = zeros(512,512,3,'single');
for channel = 1:3, color = colors(:,channel); heat(:,:,channel) = reshape(color(indices),512,512); end
overlay = uint8(round(255*(.62*repmat(base,1,1,3)+.38*heat)));
end
