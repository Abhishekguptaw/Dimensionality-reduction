clear;
Algo = 'lle_rf'
accuracy = zeros;
recall = zeros;
precision = zeros;
F_score = zeros;
Newcols = zeros;
Ncols = zeros;
for k=16
    file = sprintf('Gear%d.xlsx', k);
    X = xlsread(file);
    %% 
    data = X(:,1:end-1);
label = X(:,end);
classname = unique(label);
m = size(label,1);
Z = zeros(m,length(classname));

for i = 1:length(classname)
    dataS{i} = X(label == classname(i),:);
end

for i = 1:size(dataS,2)
    dist(i) = size(dataS{i},1);
end

dist = normalize(dist,'norm',1);
newdata = [];
for i = 1:length(classname)
    kk = size(dataS{i},1);
    n_r = floor(kk*0.1);
    index = randperm(kk,n_r);
    ndata = dataS{i}(index,:);
    newdata = [newdata;ndata];
end


    %%
    data = newdata;
    label = data(:, end);
    dataset = data(:, 1:end-1);
    
    var_data =var(dataset);
    Newcols = [];
    Ncols= [];
    
    for i = 1:size(dataset,2)
        if (var_data(1,i)> 1)
            Newcols = [Newcols i];
        end
    end
    if (Newcols>0)
        dataset = dataset(:,Newcols);
    end
    for i = 1:size(dataset,2)
        uniquevals = size(unique( dataset(:,i)),1);
        siz = size(dataset(:,i));
        if (uniquevals>(siz*.15 ))
            Ncols = [Ncols i];
        end
    end
    if (Ncols>0)
        dataset = dataset(:,Ncols);
    end
    
    
    
    X = normalize(dataset,1);
    X=X';
    K=8;
    d=12;
    [D,N] = size(X);
    [Y] = lle_dr(X,K,d);
    
    data_new = Y';
    data_new = normalize(data_new);
    [row, column ] = size(data_new);
    indices = crossvalind('Kfold',row,10);
    
    for i = 1:10
        test = (indices == i);
        train = ~test;
        mdl = fitctree(data_new(train,:), label(train,:));
        prelabel = predict(mdl, data_new(test, :));
        accuracy(i,k) = (length(find(prelabel == label(test,:)))/ length(prelabel)*100)';
        confMat = confusionmat(prelabel, label(test,:));
        %% calculation of precision, recall and F-score
        %%% recall
        for m =1:size(confMat, 1);
            recall(m)=confMat(m,m)/sum(confMat(m,:));
        end
        recall(isnan(recall))=[];
        Recall=sum(recall)/size(confMat,1);
        
        %%% précision
        for n =1:size(confMat,1);
            precision(n)=confMat(n,n)/sum(confMat(:,n));
        end
        precision(isnan(precision))=[];
        Precision=sum(precision)/size(confMat,1);
        %%% F-score
        F_score(i,k)=(100*2*Recall*Precision/(Precision+Recall))';
    end
    fprintf(file)
end
save = sprintf('accuracy_%1$s.xlsx', Algo);
xlswrite(save, accuracy);
save = sprintf('F_score_%1$s.xlsx', Algo);;
xlswrite(save, F_score);
