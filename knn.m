clear;
Algo = 'knn'
accuracy = zeros;
recall = zeros;
precision = zeros;
F_score = zeros;
Newcols = zeros;
Ncols = zeros;

for k=1:25
    file = sprintf('Gear%d.xlsx', k);
    data = xlsread(file);
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
        dataset =dataset(:,Ncols);
    end
    data = normalize(dataset,1);
    
    [row, column ] = size(dataset);
    
    indices = crossvalind('Kfold',row,10);
    
    for i = 1:10
        test = (indices == i);
        train = ~test;
        mdl = fitcknn(dataset(train,:), label(train,:),'NumNeighbors',15);
        prelabel = predict(mdl, dataset(test, :));
        accuracy(i,k) = (length(find(prelabel == label(test,:)))/ length(prelabel))';
        confMat = confusionmat(prelabel, label(test,:));
        %% calculation of precision, recall and F-score
        %%% recall
        for m =1:size(confMat, 1)
            recall(m)=confMat(m,m)/sum(confMat(m,:));
        end
        recall(isnan(recall))=[];
        Recall=sum(recall)/size(confMat,1);
        %%% précision
        for n =1:size(confMat,1)
            precision(n)=confMat(n,n)/sum(confMat(:,n));
        end
        precision(isnan(precision))=[];
        Precision=sum(precision)/size(confMat,1);
        %%% F-score
        F_score(i,k)=(2*Recall*Precision/(Precision+Recall))';
    end
    fprintf(file);
end
save = sprintf('accuracy_%1$s.xlsx', Algo);
xlswrite(save, accuracy);
save = sprintf('F_score_%1$s.xlsx', Algo);;
xlswrite(save, F_score);
