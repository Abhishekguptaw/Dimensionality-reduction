clear;
Algo = 'MC_knn';
accuracy = zeros;
recall = zeros;
precision = zeros;
F_score = zeros;
Newcols = zeros;
Ncols = zeros;

for k=1:15
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
        dataset = dataset(:,Ncols);
    end
    
    daatset = normalize(dataset,1);
       
    [row, column ] = size(dataset);
    N =row;
    data = normalize(dataset,1);
    %% Changing these values will lead to different nonlinear embeddings
    knn    = ceil(0.03*N); % each patch will only look at its knn nearest neighbors in R^d
    sigma2 = 100; % determines strength of connection in graph... see below
    
    %% now let's get pairwise distance info and create graph
    m                = size(data,1);
    dt               = squareform(pdist(data));
    [srtdDt,srtdIdx] = sort(dt,'ascend');
    dt               = srtdDt(1:knn+1,:);
    nidx             = srtdIdx(1:knn+1,:);
    
    % nz   = dt(:) > 0;
    % mind = min(dt(nz));
    % maxd = max(dt(nz));
    
    % compute weights
    tempW  = exp(-dt.^2/sigma2);
    
    % build weight matrix
    i = repmat(1:m,knn+1,1);
    W = sparse(i(:),double(nidx(:)),tempW(:),m,m);
    W = max(W,W'); % for undirected graph.
    
    % The original normalized graph Laplacian, non-corrected for density
    ld = diag(sum(W,2).^(-1/2));
    DO = ld*W*ld;
    DO = max(DO,DO');%(DO + DO')/2;
    
    % get eigenvectors
    [v,d] = eigs(DO,17,'la');
    
    eigVecIdx = nchoosek(2:4,2);
    
    data_new = normalize(v);
    
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