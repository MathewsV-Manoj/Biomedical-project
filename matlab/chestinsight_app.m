function fig = chestinsight_app()
% Native MATLAB application. It runs only MATLAB files and a MATLAB model.
root = fileparts(fileparts(mfilename('fullpath')));
modelPath = fullfile(root,'models','chestinsight_matlab_model.mat');
labels = chestinsight_labels(); currentImage = '';
fig = uifigure('Name','ChestInsight | MATLAB-only','Position',[100 80 1320 860]);
grid = uigridlayout(fig,[5 4]); grid.RowHeight={46,44,'1x','1x',68}; grid.ColumnWidth={'1x','1x','1x','1x'};
heading=uilabel(grid,'Text','ChestInsight | MATLAB-only chest X-ray analysis','FontSize',22,'FontWeight','bold'); heading.Layout.Row=1; heading.Layout.Column=[1 4];
choose= uibutton(grid,'Text','1. Choose X-ray','ButtonPushedFcn',@chooseImage); choose.Layout.Row=2; choose.Layout.Column=1;
example= uibutton(grid,'Text','Use built-in example','ButtonPushedFcn',@useExample); example.Layout.Row=2; example.Layout.Column=2;
finding=uidropdown(grid,'Items',labels,'Value','Cardiomegaly'); finding.Layout.Row=2; finding.Layout.Column=3;
run= uibutton(grid,'Text','2. Analyze in MATLAB','ButtonPushedFcn',@analyze); run.Layout.Row=2; run.Layout.Column=4;
originalAxes=uiaxes(grid); originalAxes.Layout.Row=3; originalAxes.Layout.Column=[1 2]; title(originalAxes,'Original image');
saliencyAxes=uiaxes(grid); saliencyAxes.Layout.Row=3; saliencyAxes.Layout.Column=[3 4]; title(saliencyAxes,'MATLAB saliency map');
enhancedAxes=uiaxes(grid); enhancedAxes.Layout.Row=4; enhancedAxes.Layout.Column=[1 2]; title(enhancedAxes,'MATLAB enhancement — display only');
predictionPanel=uipanel(grid,'Title','Selected finding result'); predictionPanel.Layout.Row=4; predictionPanel.Layout.Column=[3 4];
predictionGrid=uigridlayout(predictionPanel,[4 1]); predictionGrid.RowHeight={22,34,22,'1x'}; predictionGrid.Padding=[8 6 8 6];
selectedFinding=uilabel(predictionGrid,'Text','Select a disease and analyze an X-ray','FontWeight','bold'); selectedFinding.Layout.Row=1;
confidenceScore=uilabel(predictionGrid,'Text','Model confidence score: —','FontSize',21,'FontWeight','bold','FontColor',[0 .42 .45]); confidenceScore.Layout.Row=2;
decision=uilabel(predictionGrid,'Text','Threshold and result will appear here.','FontColor',[.2 .2 .2]); decision.Layout.Row=3;
scoreAxes=uiaxes(predictionGrid); scoreAxes.Layout.Row=4; title(scoreAxes,'All ten model scores'); xlabel(scoreAxes,'Score (%)');
status=uilabel(grid,'Text','Choose a de-identified frontal chest X-ray. The included MATLAB model is ready to use; retrain only when you want to replace it.','WordWrap','on'); status.Layout.Row=5; status.Layout.Column=[1 4];
    function useExample(~,~)
        currentImage=fullfile(root,'data','example_nih.png'); showInput();
    end
    function chooseImage(~,~)
        [file,folder]=uigetfile({'*.png;*.jpg;*.jpeg','Chest X-ray image (PNG/JPEG)'},'Choose a de-identified frontal chest X-ray');
        if isequal(file,0), return; end
        currentImage=fullfile(folder,file); showInput();
    end
    function showInput()
        input=imread(currentImage);
        imagesc(originalAxes,chestinsight_gray(input)); colormap(originalAxes,gray(256)); axis(originalAxes,'image','off'); title(originalAxes,'Original chest X-ray');
        status.Text='Ready. Select a finding and click Analyze in MATLAB.';
    end
    function analyze(~,~)
        if isempty(currentImage), uialert(fig,'Choose an image first.','No input'); return; end
        if ~isfile(modelPath)
            uialert(fig,'No MATLAB model is available yet. Run chestinsight_train from the MATLAB Command Window.','Train first'); return;
        end
        run.Enable='off'; choose.Enable='off'; status.Text='Running the MATLAB neural network and creating the report…'; drawnow;
        try
            result=chestinsight_predict(currentImage,finding.Value,modelPath);
            original=imread(currentImage);
            enhanced=chestinsight_enhance(original);
            overlay=chestinsight_make_overlay(original,result.saliency);
            image(saliencyAxes,overlay); axis(saliencyAxes,'image','off'); title(saliencyAxes,['MATLAB saliency: ' finding.Value]);
            imagesc(enhancedAxes,enhanced); colormap(enhancedAxes,gray(256)); axis(enhancedAxes,'image','off'); title(enhancedAxes,'MATLAB enhancement — display only');
            selectedIndex=find(strcmp(result.labels,finding.Value),1);
            scorePercent=100*result.scores(selectedIndex); thresholdPercent=100*result.thresholds(selectedIndex);
            selectedFinding.Text=['Selected finding: ' finding.Value];
        confidenceScore.Text=sprintf('Possibility / model confidence: %.1f%%',scorePercent);
            if result.flags(selectedIndex)
            decision.Text=sprintf('Result: above the research threshold (%.1f%%). Black diamond = threshold.',thresholdPercent);
                confidenceScore.FontColor=[.06 .48 .18];
            else
            decision.Text=sprintf('Result: below the research threshold (%.1f%%). Black diamond = threshold.',thresholdPercent);
                confidenceScore.FontColor=[.75 .25 .05];
            end
            drawScoreChart(scoreAxes,result.scores,result.labels,selectedIndex,result.thresholds(selectedIndex));
            output=chestinsight_export_report(result,currentImage);
            status.Text=['Complete. MATLAB-only report saved in: ' output];
        catch exception
            status.Text='Analysis failed. Read the error message and try another PNG/JPEG image.';
            uialert(fig,exception.message,'Analysis error');
        end
        run.Enable='on'; choose.Enable='on';
    end
end

function drawScoreChart(ax,scores,labels,selectedIndex,selectedThreshold)
cla(ax); values=100*scores(:); bars=barh(ax,values,'FaceColor','flat');
bars.CData=repmat([.12 .53 .57],numel(values),1); bars.CData(selectedIndex,:)=[.94 .48 .08];
yticks(ax,1:numel(labels)); yticklabels(ax,string(labels)); ax.YDir='reverse'; xlim(ax,[0 100]); grid(ax,'on');
hold(ax,'on'); plot(ax,100*selectedThreshold,selectedIndex,'kd','MarkerFaceColor','k','MarkerSize',6); hold(ax,'off');
title(ax,'All ten model scores — orange is selected'); xlabel(ax,'Model score (%)');
end
