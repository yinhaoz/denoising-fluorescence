function coefMatrix = NN_BP(Data,D,maxCoef,coefMatrix)
MulFactor = 5;
numIterations = 15;
lambda = 1e-9;

numberOfIncreasedElems = ceil(maxCoef);
maxValues = max(coefMatrix) - min(coefMatrix)/2;
% we increase 'numberOfIncreasedElems' values only from the initial coefficient values.
for col = 1:size(coefMatrix,2)
    I = find(coefMatrix(:,col)==0);
    if (length(I) > numberOfIncreasedElems)
        permutation = randperm(size(coefMatrix,1)-maxCoef);
        coefMatrix(I(permutation(1:numberOfIncreasedElems)),col) = maxValues(col);
    else
        permutation = randperm(size(coefMatrix,1));
        coefMatrix(permutation(1:numberOfIncreasedElems),col) = maxValues(col);
    end
end
for iter = 1:numIterations
    coefMatrix = coefMatrix.*(D'*Data)./(D'*D*coefMatrix+lambda);
end

for s = 1:size(Data,2)
    [v,i] = sort(coefMatrix(:,s)); 
    FirstNonZeroIdx = size(coefMatrix,1)-maxCoef+1;
    coefMatrix(i(1:FirstNonZeroIdx-1),s) = 0;
    coefMatrix(i(FirstNonZeroIdx:end),s) = lsqnonneg(D(:,i(FirstNonZeroIdx:end)),Data(:,s),coefMatrix(i(FirstNonZeroIdx:end),s));
end