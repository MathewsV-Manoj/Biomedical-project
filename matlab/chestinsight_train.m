function model = chestinsight_train(varargin)
% Train and evaluate a MATLAB-only multi-label neural network.
% No Python or Deep Learning Toolbox is required.
parser = inputParser;
addParameter(parser,'MaxEpochs',10,@(x)isscalar(x) && x>=1);
addParameter(parser,'BatchSize',64,@(x)isscalar(x) && x>=1);
addParameter(parser,'LearningRate',1e-3,@(x)isscalar(x) && x>0);
addParameter(parser,'HiddenSizes',[64 32],@(x)isnumeric(x) && numel(x)==2);
addParameter(parser,'ForceRebuildFeatures',false,@islogical);
parse(parser,varargin{:}); options = parser.Results;
root = fileparts(fileparts(mfilename('fullpath')));
rng(42,'twister');
data = chestinsight_load_data(options.ForceRebuildFeatures);
train = find(data.splits == "train"); validation = find(data.splits == "val"); test = find(data.splits == "test");
assert(isempty(intersect(data.patientIds(train),data.patientIds(validation))),'Patient leakage between train and validation.');
assert(isempty(intersect(data.patientIds(train),data.patientIds(test))),'Patient leakage between train and test.');
assert(isempty(intersect(data.patientIds(validation),data.patientIds(test))),'Patient leakage between validation and test.');
labels = data.labels; classCount = numel(labels); inputCount = size(data.features,1);
hidden1 = options.HiddenSizes(1); hidden2 = options.HiddenSizes(2);
network = struct('W1',single(randn(hidden1,inputCount)*sqrt(2/inputCount)), ...
    'b1',zeros(hidden1,1,'single'),'W2',single(randn(hidden2,hidden1)*sqrt(2/hidden1)), ...
    'b2',zeros(hidden2,1,'single'),'W3',single(randn(classCount,hidden2)*sqrt(2/hidden2)), ...
    'b3',zeros(classCount,1,'single'));
positives = sum(data.targets(:,train),2);
positiveWeights = min((numel(train)-positives)./max(positives,1),20);
moments = initializeMoments(network); bestAuc = -Inf; bestNetwork = network; history = struct([]); step = 0;
for epoch = 1:options.MaxEpochs
    ordering = train(randperm(numel(train))); totalLoss = 0;
    for first = 1:options.BatchSize:numel(ordering)
        batch = ordering(first:min(first+options.BatchSize-1,numel(ordering)));
        x = data.features(:,batch); y = data.targets(:,batch);
        [probabilities,cache] = forward(network,x);
        weightMask = 1 + (positiveWeights-1).*y;
        loss = -mean(sum(weightMask.*y.*log(max(probabilities,eps('single'))) + (1-y).*log(max(1-probabilities,eps('single'))),1));
        gradients = backward(network,cache,probabilities,y,positiveWeights);
        step = step + 1; [network,moments] = adamUpdate(network,gradients,moments,step,options.LearningRate);
        totalLoss = totalLoss + double(loss)*numel(batch);
    end
    validationScores = predictBatch(network,data.features(:,validation));
    validationAuc = macroAuc(data.targets(:,validation),validationScores);
    history(epoch).epoch = epoch; history(epoch).trainingLoss = totalLoss/numel(train); history(epoch).validationMacroAuc = validationAuc; %#ok<AGROW>
    fprintf('Epoch %d/%d | loss %.4f | validation macro AUROC %.3f\n',epoch,options.MaxEpochs,history(epoch).trainingLoss,validationAuc);
    if validationAuc > bestAuc, bestAuc = validationAuc; bestNetwork = network; bestEpoch = epoch; end
end
network = bestNetwork;
validationScores = predictBatch(network,data.features(:,validation)); thresholds = selectThresholds(data.targets(:,validation),validationScores);
testScores = predictBatch(network,data.features(:,test)); metrics = calculateMetrics(data.targets(:,test),testScores,thresholds,labels);
model = struct('name','MATLAB-only multi-label neural network','network',network,'labels',{labels}, ...
    'thresholds',thresholds,'imageSize',32,'provenance',struct( ...
    'implementation','MATLAB-only custom two-hidden-layer ReLU network', ...
    'trainingPatients',numel(unique(data.patientIds(train))), ...
    'validationPatients',numel(unique(data.patientIds(validation))), ...
    'testPatients',numel(unique(data.patientIds(test))),'bestEpoch',bestEpoch, ...
    'options',options,'limitation','Pilot subset with report-derived labels; research use only.'));
modelsDir = fullfile(root,'models'); reportsDir = fullfile(root,'reports','matlab_only');
if ~isfolder(modelsDir), mkdir(modelsDir); end
if ~isfolder(reportsDir), mkdir(reportsDir); end
save(fullfile(modelsDir,'chestinsight_matlab_model.mat'),'model','-v7.3');
historyTable = struct2table(history); writetable(historyTable,fullfile(reportsDir,'history.csv'));
writetable(metrics.table,fullfile(reportsDir,'test_metrics.csv'));
evaluation = struct('macroAuc',metrics.macroAuc,'macroAveragePrecision',metrics.macroAveragePrecision, ...
    'macroF1',metrics.macroF1,'bestEpoch',bestEpoch,'labels',{labels},'thresholds',thresholds, ...
    'note','Thresholds selected on validation patients only; test patients were held out.');
writeText(fullfile(reportsDir,'evaluation.json'),jsonencode(evaluation,PrettyPrint=true));
createFigures(historyTable,metrics,labels,reportsDir);
fprintf('MATLAB-only training complete. Test macro AUROC %.3f.\n',metrics.macroAuc);
end

function [p,cache] = forward(network,x)
cache.x = x; cache.z1 = network.W1*x+network.b1; cache.a1=max(cache.z1,0);
cache.z2 = network.W2*cache.a1+network.b2; cache.a2=max(cache.z2,0);
cache.z3 = network.W3*cache.a2+network.b3; p=sigmoid(cache.z3);
end
function p = predictBatch(network,x), p=forward(network,x); end
function gradients = backward(network,cache,p,y,positiveWeights)
batch = size(y,2); mask = 1+(positiveWeights-1).*y; dz3=((p-y).*mask)/batch;
gradients.W3=dz3*cache.a2'; gradients.b3=sum(dz3,2);
da2=network.W3'*dz3; dz2=da2.*single(cache.z2>0);
gradients.W2=dz2*cache.a1'; gradients.b2=sum(dz2,2);
da1=network.W2'*dz2; dz1=da1.*single(cache.z1>0);
gradients.W1=dz1*cache.x'; gradients.b1=sum(dz1,2);
end
function moments = initializeMoments(network)
fields=fieldnames(network); for i=1:numel(fields), f=fields{i}; moments.m.(f)=zeros(size(network.(f)),'like',network.(f)); moments.v.(f)=zeros(size(network.(f)),'like',network.(f)); end
end
function [network,moments] = adamUpdate(network,gradients,moments,step,learningRate)
beta1=.9; beta2=.999; fields=fieldnames(network);
for i=1:numel(fields)
    f=fields{i}; g=gradients.(f); moments.m.(f)=beta1*moments.m.(f)+(1-beta1)*g; moments.v.(f)=beta2*moments.v.(f)+(1-beta2)*(g.^2);
    mhat=moments.m.(f)/(1-beta1^step); vhat=moments.v.(f)/(1-beta2^step);
    network.(f)=network.(f)-learningRate*mhat./(sqrt(vhat)+1e-8);
end
end
function values = sigmoid(values), values=1./(1+exp(-max(min(values,30),-30))); end
function auc = macroAuc(y,p), values=zeros(size(y,1),1); for i=1:size(y,1), values(i)=aucOne(y(i,:),p(i,:)); end, auc=mean(values,'omitnan'); end
function auc = aucOne(y,p)
[~,order]=sort(p,'descend'); y=y(order); positive=sum(y); negative=numel(y)-positive;
if positive==0 || negative==0, auc=NaN; return; end
tpr=[0 cumsum(y)/positive 1]; fpr=[0 cumsum(1-y)/negative 1]; auc=trapz(fpr,tpr);
end
function thresholds = selectThresholds(y,p)
thresholds=zeros(size(y,1),1); candidates=linspace(.01,.99,99);
for i=1:size(y,1)
    scores=zeros(size(candidates));
    for c=1:numel(candidates)
        predicted=p(i,:)>=candidates(c); tp=sum(predicted & y(i,:)); fp=sum(predicted & ~y(i,:)); fn=sum(~predicted & y(i,:));
        scores(c)=2*tp/max(2*tp+fp+fn,1);
    end
    [~,best]=max(scores); thresholds(i)=candidates(best);
end
end
function metrics = calculateMetrics(y,p,t,labels)
n=numel(labels); auc=zeros(n,1); ap=zeros(n,1); precision=zeros(n,1); recall=zeros(n,1); specificity=zeros(n,1); f1=zeros(n,1); positives=sum(y,2); tn=zeros(n,1); fp=tn; fn=tn; tp=tn;
for i=1:n
    auc(i)=aucOne(y(i,:),p(i,:)); ap(i)=averagePrecision(y(i,:),p(i,:)); predicted=p(i,:)>=t(i);
    tp(i)=sum(predicted & y(i,:)); fp(i)=sum(predicted & ~y(i,:)); fn(i)=sum(~predicted & y(i,:)); tn(i)=sum(~predicted & ~y(i,:));
    precision(i)=tp(i)/max(tp(i)+fp(i),1); recall(i)=tp(i)/max(tp(i)+fn(i),1); specificity(i)=tn(i)/max(tn(i)+fp(i),1); f1(i)=2*tp(i)/max(2*tp(i)+fp(i)+fn(i),1);
end
metrics.table=table(string(labels(:)),positives,auc,ap,t,tn,fp,fn,tp,precision,recall,specificity,f1,'VariableNames',{'finding','positives','auroc','average_precision','threshold','tn','fp','fn','tp','precision','recall','specificity','f1'});
metrics.macroAuc=mean(auc,'omitnan'); metrics.macroAveragePrecision=mean(ap,'omitnan'); metrics.macroF1=mean(f1,'omitnan');
end
function ap = averagePrecision(y,p), [~,order]=sort(p,'descend'); y=y(order); ap=sum((cumsum(y)./(1:numel(y))).*y)/max(sum(y),1); end
function createFigures(history,metrics,labels,folder)
fig=figure('Visible','off','Color','w','Position',[100 100 1200 450]); tiledlayout(1,2,'Padding','compact');
nexttile; plot(history.epoch,history.trainingLoss,'o-','LineWidth',1.5); xlabel('Epoch'); ylabel('Weighted binary cross-entropy'); title('MATLAB-only training loss'); grid on;
nexttile; bar(metrics.table.auroc); ylim([0 1]); xticks(1:numel(labels)); xticklabels(labels); xtickangle(45); ylabel('AUROC'); title('Held-out pilot test performance'); grid on;
exportgraphics(fig,fullfile(folder,'training_and_auroc.png'),'Resolution',160); close(fig);
end
function writeText(path,text), file=fopen(path,'w'); assert(file~=-1,'Cannot write output.'); cleanup=onCleanup(@() fclose(file)); fprintf(file,'%s',text); end
