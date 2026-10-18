function outputDirectory = chestinsight_export_report(result, imagePath, outputDirectory)
% Save a MATLAB-only analysis bundle: PNG, MAT, JSON, and CSV.
if nargin < 3 || isempty(outputDirectory)
    root = fileparts(fileparts(mfilename('fullpath')));
    outputDirectory = fullfile(root,'reports','matlab_only_runs',char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
end
if ~isfolder(outputDirectory), mkdir(outputDirectory); end
original = imread(imagePath);
enhanced = enhance(original);
overlay = chestinsight_make_overlay(original,result.saliency);
tableResults = table(string(result.labels(:)),result.scores(:),result.thresholds(:),string(result.flags(:)), ...
    'VariableNames',{'Finding','ModelScore','Threshold','AboveThreshold'});
writetable(tableResults,fullfile(outputDirectory,'findings.csv'));
save(fullfile(outputDirectory,'matlab_result.mat'),'result','tableResults','original','enhanced','overlay','-v7.3');
summary = struct('model',result.modelName,'findingExplained',result.finding,'labels',{result.labels}, ...
    'scores',result.scores,'thresholds',result.thresholds,'aboveThreshold',result.flags, ...
    'notice',result.notice,'created',char(datetime('now')));
writeText(fullfile(outputDirectory,'analysis.json'),jsonencode(summary,PrettyPrint=true));
fig=figure('Visible','off','Color','w','Position',[100 100 1500 900]); tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
ax=nexttile; imagesc(ax,chestinsight_gray(original)); axis(ax,'image','off'); colormap(ax,gray(256)); title(ax,'Original chest X-ray');
ax=nexttile; imagesc(ax,enhanced); axis(ax,'image','off'); colormap(ax,gray(256)); title(ax,'MATLAB contrast enhancement (display only)');
ax=nexttile; image(ax,overlay); axis(ax,'image','off'); title(ax,['MATLAB input-gradient saliency: ' result.finding],'Interpreter','none');
ax=nexttile; [sorted,index]=sort(result.scores,'ascend'); barh(ax,sorted,'FaceColor',[.05 .52 .56]); yticks(ax,1:numel(index)); yticklabels(ax,string(result.labels(index))); xlim(ax,[0 1]); xlabel(ax,'Independent model score'); title(ax,'Ten multi-label scores'); grid(ax,'on');
ax=nexttile; histogram(ax,double(original(:)),0:256,'FaceColor',[.2 .35 .65],'EdgeColor','none'); title(ax,'MATLAB image histogram'); xlabel(ax,'Grayscale intensity'); ylabel(ax,'Pixel count');
ax=nexttile; axis(ax,'off'); text(ax,0,1,{result.modelName,'',result.notice,'','Saliency marks input pixels that most change the selected model output.','It is not a lesion boundary.'},'VerticalAlignment','top','FontSize',10,'Interpreter','none');
sgtitle(fig,'ChestInsight | MATLAB-only multi-label X-ray analysis','FontWeight','bold');
exportgraphics(fig,fullfile(outputDirectory,'matlab_only_report.png'),'Resolution',160); close(fig);
imwrite(enhanced,fullfile(outputDirectory,'enhanced.png')); imwrite(overlay,fullfile(outputDirectory,'saliency_overlay.png'));
end

function enhanced = enhance(image)
enhanced=chestinsight_enhance(image);
end
function writeText(path,text), file=fopen(path,'w'); assert(file~=-1,'Cannot write output.'); cleanup=onCleanup(@() fclose(file)); fprintf(file,'%s',text); end
