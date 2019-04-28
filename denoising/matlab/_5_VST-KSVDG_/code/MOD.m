function [Dictionary,output] = MOD(Data,param)
% =========================================================================
%                          MOD algorithm
% =========================================================================
% Given for comparison reasons only. For detils please see the paper
% "Method of optimal directions for frame design", written by K. Engan, 
% S.O. Aase, and J.H. Husfy, appeared in the IEEE International Conference 
% on Acoustics, Speech, and Signal Processing, 1999.
% =========================================================================
% INPUT ARGUMENTS:
% Data                         an nXN matrix that contins N signals (Y), each of dimension n. 
% param                        structure that includes all required
%                                 parameters for the K-SVD execution.
%                                 Required fields are:
%    K, ...                    the number of dictionary elements to train
%    numIteration,...          number of iterations to perform.
%    errorFlag...              if =0, a fix number of coefficients is
%                                 used for representation of each signal. If so, param.L must be
%                                 specified as the number of representing atom. if =1, arbitrary number
%                                 of atoms represent each signal, until a specific representation error
%                                 is reached. If so, param.errorGoal must be specified as the allowed
%                                 error.
%    (optional, see errorFlag) L,...                 % maximum coefficients to use in OMP coefficient calculations.
%    (optional, see errorFlag) errorGoal, ...        % allowed representation error in representing each signal.
%    InitializationMethod,...  mehtod to initialize the dictionary, can
%                                 be one of the following arguments: 
%                                 * 'DataElements' (initialization by the signals themselves), or: 
%                                 * 'GivenMatrix' (initialization by a given matrix param.initialDictionary).
%    (optional, see InitializationMethod) initialDictionary,...      % if the initialization method 
%                                 is 'GivenMatrix', this is the matrix that will be used.
%    preserveDCAtom, ...       =1 for a DC atom (in which all entries are equal) to be generated, and
%                                 not changed throughout the training.
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
%                                param.displayProgress=1 and
%                                param.errorFlag = 0)
%    numCoef                     A vector of length param.numIteration that
%                                include the average number of coefficients required for representation
%                                of each signal (in each iteration) (defined only if
%                                param.displayProgress=1 and
%                                param.errorFlag = 1)
% =========================================================================



if (size(Data,2) < param.K)
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));
    return;
elseif (strcmp(param.InitializationMethod,'DataElements'))
    Dictionary(:,1:param.K) = Data(:,1:param.K);
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
    Dictionary = param.initialDictionary;
end
%normalize the dictionary.
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.
K = size(Dictionary,2);
totalErr = zeros(1,param.numIteration);

if (size(param.TrueDictionary)==size(Dictionary))
    displayErrorWithTrueDictionary = 1;
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);
    ratio = zeros(param.numIteration+1,1);
else
    displayErrorWithTrueDictionary = 0;
end
numCoef = param.L;

for iterNum = 1:param.numIteration
    % find the coefficients
    if (param.errorFlag==0)
        %CoefMatrix = mexOMPIterative2(Data, [FixedDictionaryElement,Dictionary],param.L);
        CoefMatrix = OMP(Dictionary,Data, param.L);
    else
        %CoefMatrix = mexOMPerrIterative(Data, [FixedDictionaryElement,Dictionary],param.errorGoal);
        CoefMatrix = OMPerr(Dictionary,Data, param.errorGoal);
        param.L = 1;
    end
    % improve the dictionary
    Dictionary = Data*CoefMatrix'*inv(CoefMatrix*CoefMatrix' + 1e-7*speye(size(CoefMatrix,1)));
    sumDictElems = sum(abs(Dictionary));
    zerosIdx = find(sumDictElems<eps);
    Dictionary(:,zerosIdx) = randn(size(Dictionary,1),length(zerosIdx));
    Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
    if (iterNum>1 & param.displayProgress)
        if (param.errorFlag==0)
            output.totalerr(iterNum-1) = sqrt(sum(sum((Data-Dictionary*CoefMatrix).^2))/prod(size(Data)));
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
end
output.CoefMatrix = CoefMatrix;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanseBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanseBetweenDictionaries(original,new)
% first, all the column in oiginal starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end
for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [minValue,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);


    

    




