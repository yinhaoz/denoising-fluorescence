function [Dictionary,output] = KSVD_NN(...
    Data,... % an nXN matrix that contins N signals (Y), each of dimension n.
    param)
% =========================================================================
%                        Non Negative K-SVD algorithm
% =========================================================================
% The NN-K-SVD algorithm finds a non-negative dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can sparsely represent each signal. Detailed discussion on the algorithm
% and possible applications can be found in "K-SVD and its non-negative 
% variant for dictionary design", written by M. Aharon, M. Elad, and A.M. Bruckstein 
% and appeared in the Proceedings of the SPIE conference wavelets, Vol.
% 5914, July 2005. 
% =========================================================================
% INPUT ARGUMENTS:
% Data                         an nXN matrix that contins N signals (Y), each of dimension n. 
% param                        structure that includes all required
%                                 parameters for the K-SVD execution.
%                                 Required fields are:
%    K, ...                    the number of dictionary elements to train
%    numIteration,...          number of iterations to perform.
%    L,...                     maximum coefficients to use in OMP coefficient calculations.
%    InitializationMethod,...  mehtod to initialize the dictionary, can
%                                 be one of the following arguments: 
%                                 * 'DataElements' (initialization by the signals themselves), or: 
%                                 * 'GivenMatrix' (initialization by a given matrix param.initialDictionary).
%    (optional, see InitializationMethod) initialDictionary,...      % if the initialization method 
%                                 is 'GivenMatrix', this is the matrix that will be used.
%    (optional) TrueDictionary, ...        % if specified, in each
%                                 iteration the difference between this dictionary and the trained one
%                                 is measured and displayed.
%    displayProgress, ...      if =1 progress information is displyed. If param.errorFlag==0, 
%                                 the average repersentation error (RMSE) is displayed, while if 
%                                 param.errorFlag==1, the average number of required coefficients for 
%                                 representation of each signal is displayed.
% =========================================================================
% OUTPUT ARGUMENTS:
%  Dictionary                  The extracted dictionary of size nX(param.K).
%  output                      Struct that contains information about the current run. It may include the following fields:
%    CoefMatrix                  The final coefficients matrix (it should hold that Data equals approximately Dictionary*output.CoefMatrix.
%    ratio                       If the true dictionary was defined (in
%                                synthetic experiments), this parameter holds a vector of length
%                                param.numIteration that includes the detection ratios in each
%                                iteration).
%    totalerr                    The total representation error after each
%                                iteration (defined only if
%                                param.displayProgress=1)
% =========================================================================

if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end
totalerr(1) = 99999;
if (isfield(param,'errorFlag')==0)
    param.errorFlag = 0;
end

%Data(Data<0) = 0;

if (isfield(param,'TrueDictionary'))
    displayErrorWithTrueDictionary = 1;
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);
    ratio = zeros(param.numIteration+1,1);
else
    displayErrorWithTrueDictionary = 0;
	ratio = 0;
end
if (param.preserveDCAtom>0)
    FixedDictionaryElement(:,1) = 1/sqrt(size(Data,1));
else
    FixedDictionaryElement = [];
end
% coefficient calculation method is OMP with fixed number of coefficients

if (size(Data,2) < param.K)
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));
    return;
elseif (strcmp(param.InitializationMethod,'DataElements'))
    Dictionary(:,1:param.K) = Data(:,1:param.K-param.preserveDCAtom);
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
    Dictionary = param.initialDictionary(:,1:param.K-param.preserveDCAtom);
end
% reduce the components in Dictionary that are spanned by the fixed
% elements
if (param.preserveDCAtom)
    tmpMat = FixedDictionaryElement \ Dictionary;
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;
end
%normalize the dictionary.
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
totalErr = zeros(1,param.numIteration);

CoefMatrix = sparse(param.K,size(Data,2));

% the K-SVD algorithm starts here.
for iterNum = 1:param.numIteration
    % find the coefficients
	CoefMatrix = NN_BP(Data, [FixedDictionaryElement,Dictionary],param.L,CoefMatrix);
    
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2));
    for j = rPerm
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data,...
            [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
            CoefMatrix ,param.L);
        Dictionary(:,j) = betterDictionaryElement;
        if (param.preserveDCAtom)
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;
    end

    if (iterNum>1 & param.displayProgress)
        if (param.errorFlag==0)
            output.totalerr(iterNum-1) = sqrt(sum(sum((Data-[FixedDictionaryElement,Dictionary]*CoefMatrix).^2))/prod(size(Data)));
            disp(['Iteration   ',num2str(iterNum),'   Total error is: ',num2str(output.totalerr(iterNum-1))]);
        else
            output.numCoef(iterNum-1) = length(find(CoefMatrix))/size(Data,2);
            disp(['Iteration   ',num2str(iterNum),'   Average number of coefficients: ',num2str(output.numCoef(iterNum-1))]);
        end
    end
    if (displayErrorWithTrueDictionary ) 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanseBetweenDictionaries(param.TrueDictionary,Dictionary);
        disp(strcat(['Iteration  ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));
        output.ratio = ratio;
    end
    Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data);
    
    if (isfield(param,'waitBarHandle'))
        waitbar(iterNum/param.counterForWaitBar);
    end
end

output.CoefMatrix = CoefMatrix;
Dictionary = [FixedDictionaryElement,Dictionary];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Dictionary,j,CoefMatrix,numCoefUsed)
if (length(who('numCoefUsed'))==0)
    numCoefUsed = 1;
end
relevantDataIndices = find(CoefMatrix(j,:)); % the data indices that uses the j'th dictionary element.
if (length(relevantDataIndices)<1) %(length(relevantDataIndices)==0)
    ErrorMat = Data-Dictionary*CoefMatrix;
    ErrorNormVec = sum(ErrorMat.^2);
    [d,i] = max(ErrorNormVec);
    betterDictionaryElement = Data(:,i);%ErrorMat(:,i); %
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));
    CoefMatrix(j,:) = 0;
    NewVectorAdded = 1;
    return;
end

NewVectorAdded = 0;
reduced_coeff = CoefMatrix(:, relevantDataIndices);
reduced_Data = Data (:, relevantDataIndices);
saveDebugDict = Dictionary(:,j);
saveDebugCoef = reduced_coeff(j,:);
%debug1 = sum(sum((reduced_Data - Dictionary*reduced_coeff).^2));
reduced_coeff (j, :) = 0;    % all but the j-th element

err_mat = reduced_Data - Dictionary * reduced_coeff;
[U S V flag] = svds((err_mat), 1);


% check for sign, flip U and V's sign if negative.
Idx_U = find(U<0);
Idx_V = find(V<0);

u1 = U; u1(Idx_U) = 0; v1 = V; v1(Idx_V) = 0;approx1 = norm(err_mat- u1*v1'*S);
u1 = zeros(size(U)); u1(Idx_U) = -U(Idx_U); v1 = zeros(size(V)); v1(Idx_V) = -V(Idx_V);approx2 = norm(err_mat- u1*v1'*S);
if (approx1<= approx2)
	betterDictionaryElement = U;
	betterDictionaryElement(Idx_U) = 0;
	coefs = V;
	coefs(Idx_V) = 0;
else
	betterDictionaryElement = zeros(size(U));
	betterDictionaryElement(Idx_U) = -U(Idx_U);
	coefs = zeros(size(V));
	coefs(Idx_V) = -V(Idx_V);
end

newAtomNorm = sqrt(betterDictionaryElement'*betterDictionaryElement);
betterDictionaryElement = betterDictionaryElement/newAtomNorm;
coefs = coefs * newAtomNorm;
% coefs(coefs<0) = 0;

newE = sum(sum(((reduced_Data - Dictionary(:,[1:j-1,j+1:end])*reduced_coeff([1:j-1,j+1:end],:))-betterDictionaryElement*coefs').^2));
oldE = sum(sum(((reduced_Data - Dictionary(:,[1:j-1,j+1:end])*reduced_coeff([1:j-1,j+1:end],:))-saveDebugDict*saveDebugCoef).^2));
if (newE>oldE)
	for iter = 1:30 % the number of iterations
		betterDictionaryElement = err_mat*coefs/(coefs'*coefs);
		betterDictionaryElement(betterDictionaryElement<0) = 0;
		coefs = err_mat'*betterDictionaryElement/(betterDictionaryElement'*betterDictionaryElement);
		coefs(coefs<0) = 0;
	end
	newAtomNorm = sqrt(betterDictionaryElement'*betterDictionaryElement);
	betterDictionaryElement = betterDictionaryElement/newAtomNorm;
	coefs = coefs * newAtomNorm;
	reduced_coeff(j,:) =  coefs;
end
reduced_coeff(j,:) =  coefs;
CoefMatrix (:, relevantDataIndices) =reduced_coeff;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new)
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = new(:,i);
end
for i = 1:size(original,2)
    d = original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [minValue,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  I_clearDictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.999;
T1 = 3;
K=size(Dictionary,2);
Er=sum((Data-Dictionary*CoefMatrix).^2,1); % remove identical atoms
G=Dictionary'*Dictionary; G = G-diag(diag(G));
for jj=1:1:K,
    if max(G(jj,:))>T2 | length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 ,
        [val,pos]=max(Er);
        Er(pos(1))=0;
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));
        G=Dictionary'*Dictionary; G = G-diag(diag(G));
    end;
end;

    

    




