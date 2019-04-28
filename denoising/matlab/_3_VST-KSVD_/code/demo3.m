function demo3()
% NN-KSVD running file - a synthetic test.
% in this file a synthetic test of the NN-K-SVD (non-negative K-SVD)
% algorithm is performed. This exact Test is presented in "K-SVD and its non-negative 
% variant for dictionary design", written by M. Aharon, M. Elad, and A.M. Bruckstein 
% and appeared in the Proceedings of the SPIE conference wavelets, Vol.
% 5914, July 2005. 

param.K = 10*9;
param.L = 5;
 
SNR = 20;
numRequiredPoints = 2000;
sizeOfElem = 64;
A = zeros(sizeOfElem,90);
param.numIteration = 200;
param.initialDictionary = rand(sizeOfElem,param.K);
param.InitializationMethod = 'GivenMatrix';

% =================================
% end of parameter setting
% =================================

basisFunction{1} = [0 0 0 0 0 0 0 0 ;
                                              0 0 0 1 1 0 0 0 ;
                                              0 0 1 0 1 0 0 0;
                                              0 0 1 0 1 0 0 0;
                                              0 0 0 0 1 0 0 0 ;
                                              0 0 0 0 1 0 0 0 ;
                                              0 0 0 0 1 0 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{2} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 1 0 0 1 0 0;
                                              0 0 0 0 1 1 0 0;
                                              0 0 0 1 1 0 0 0 ;
                                              0 0 1 1 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{3} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 1 1 0 0;
                                              0 0 0 1 1 0 0 0;
                                              0 0 0 0 1 1 0 0 ;
                                              0 0 0 0 0 1 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{4} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 0 0 1 0 0 ;
                                              0 0 1 0 0 1 0 0;
                                              0 0 1 0 0 1 0 0;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 1 0 0 ;
                                              0 0 0 0 0 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{5} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 1 0 0 0 0 0;
                                              0 0 1 1 1 1 0 0;
                                              0 0 0 0 1 1 0 0 ;
                                              0 0 0 0 1 1 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{6} = [0 0 0 0 0 0 0 0 ;
                                              0 0 0 1 1 1 0 0 ;
                                              0 0 1 0 0 0 0 0;
                                              0 0 1 0 0 0 0 0;
                                              0 0 1 0 1 1 0 0 ;
                                              0 0 1 0 0 1 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{7} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 1 0 0;
                                              0 0 0 0 1 1 0 0;
                                              0 0 0 1 1 0 0 0 ;
                                              0 0 0 1 0 0 0 0 ;
                                              0 0 0 1 0 0 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{8} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 1 0 0 1 0 0;
                                              0 0 1 1 1 1 0 0;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 1 0 0 1 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
 basisFunction{9} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 1 0 0 1 0 0;
                                              0 0 1 1 1 1 0 0;
                                              0 0 0 0 0 1 0 0 ;
                                              0 0 0 0 0 1 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
basisFunction{10} = [0 0 0 0 0 0 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 1 0 0 1 0 0;
                                              0 0 1 0 0 1 0 0;
                                              0 0 1 0 0 1 0 0 ;
                                              0 0 1 0 0 1 0 0 ;
                                              0 0 1 1 1 1 0 0 ;
                                              0 0 0 0 0 0 0 0 ];
counter = 1;
for i=1:10
    d = basisFunction{i};
    d1 = [zeros(8,2),d(:,1:6)];
    d2 = [d(:,3:end),zeros(8,2)];
    d3 = [zeros(1,8); d(1:7,:)];
    d4 = [d(1:7,:); zeros(1,8)];
    d5 = [zeros(1,8); d1(1:7,:)];
    d6 = [zeros(1,8); d2(1:7,:)];
    d7 = [d1(1:7,:); zeros(1,8)];
    d8 = [d2(1:7,:); zeros(1,8)];
    d = d(:);d1 = d1(:); d2 = d2(:);d3 = d3(:);d4 = d4(:);d5 = d5(:);d6 = d6(:);d7 = d7(:);d8 = d8(:);
    A(:,counter) = d/(d'*d);
    A(:,counter+1) = d1/(d1'*d1);
    A(:,counter+2) = d2/(d2'*d2);
    A(:,counter+3) = d3/(d3'*d3);
    A(:,counter+4) = d4/(d4'*d4);
    A(:,counter+5) = d5/(d5'*d5);
    A(:,counter+6) = d6/(d6'*d6);
    A(:,counter+7) = d7/(d7'*d7);
    A(:,counter+8) = d8/(d8'*d8);
    counter = counter+9;
end
positiveFlag = 1; 

[D,xOrig] = CreateDataFromDictionarySimple(A, numRequiredPoints,param.L,...
     positiveFlag);
 
 noise = randn(size(D));
if (SNR==0)
    D = D;
else
    actualNoise = calcNoiseFromSNR(SNR,D, noise);
    D =  D + actualNoise;
end

A = A./repmat(sqrt(diag(A'*A)'),64,1);

displayDictionaryElementsAsImage(A, 10, 9,8,8,0);
title('Original dictionary');

param.TrueDictionary = A;
param.preserveDCAtom = 0;
param.displayProgress = 1;

%=============================================
% Run the NN-KSVD function
%=============================================
[Dictionary,output] = KSVD_NN(D,param);

%=============================================
% display the results
%=============================================
Dictionary = Dictionary./repmat(sqrt(diag(Dictionary'*Dictionary)'),64,1);
figure(1)
displayDictionaryElementsAsImage(Dictionary,10, 9,8,8,0);
title(['Found dictionary']);
figure(2);



function [D,xOrig] = CreateDataFromDictionarySimple(dictionary, numElements, numCoef,positiveFlag)
maxRangeOfCoef = 1;
resolution = 0.0001;

xOrig = zeros(size(dictionary,2),numElements);
coefs = randn(numCoef,numElements)*maxRangeOfCoef;
if (positiveFlag)
    idx = find(coefs<0);
    coefs(idx) =  coefs(idx)*-1;
end
xOrig(1:numCoef,:) = coefs;
for i=1:size(xOrig,2)
    xOrig(:,i) = xOrig(randperm(size(xOrig,1)),i);
end
%dictionaryElementIndices = randsrc(numCoef*numElements,1,[1:size(dictionary,2)])   ; 
%matrixOfIndices = repmat([1:numElements],numCoef,1);
%xOrig(sub2ind(size(xOrig),dictionaryElementIndices,matrixOfIndices(:))) = coefs;
D = dictionary*xOrig;

function actualNoise = calcNoiseFromSNR(TargerSNR, signal, randomNoise)
signal = signal(:);
randomNoiseRow = randomNoise(:);
signal_2 = sum(signal.^2);
ActualNoise_2 = signal_2/(10^(TargerSNR/10));
noise_2 = sum(randomNoiseRow.^2);
ratio = ActualNoise_2./noise_2;
actualNoise = randomNoiseRow.*repmat(sqrt(ratio),size(randomNoiseRow,1),1);
actualNoise = reshape(actualNoise,size(randomNoise));
