function [Dictionary, data, coefs] = gererateSyntheticDictionaryAndData(N, L, dim, K, SNRdB)


randn('state',sum(100*clock));
rand('state',sum(100*clock));

Dictionary = randn(dim,K);
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));

[data,coefs] = CreateDataFromDictionarySimple(Dictionary, N, L);

if (SNRdB==0) | (SNRdB == 80) 
    return
else
    noise = randn(size(data));
    actualNoise = calcNoiseFromSNR(SNRdB,data, noise);
    SNR = calcSNR(data, data+actualNoise);
    data =  data + actualNoise*SNR/SNRdB;   
end

function [D,xOrig] = CreateDataFromDictionarySimple(dictionary, numElements, numCoef)
maxRangeOfCoef = 1;
resolution = 0.0001;

xOrig = zeros(size(dictionary,2),numElements);
%vecOfValues = -1*maxRangeOfCoef:resolution:maxRangeOfCoef;
%coefs = randsrc(numCoef,numElements,vecOfValues);
coefs = randn(numCoef,numElements)*maxRangeOfCoef;
xOrig(1:numCoef,:) = coefs;
for i=1:size(xOrig,2)
    xOrig(:,i) = xOrig(randperm(size(xOrig,1)),i);
end
%dictionaryElementIndices = randsrc(numCoef*numElements,1,[1:size(dictionary,2)])   ; 
%matrixOfIndices = repmat([1:numElements],numCoef,1);
%xOrig(sub2ind(size(xOrig),dictionaryElementIndices,matrixOfIndices(:))) = coefs;
D = dictionary*xOrig;

function  actualNoise = calcNoiseFromSNR(TargerSNR, signal, randomNoise)
signal = signal(:);
randomNoiseRow = randomNoise(:);
signal_2 = sum(signal.^2);
ActualNoise_2 = signal_2/(10^(TargerSNR/10));
noise_2 = sum(randomNoiseRow.^2);
ratio = ActualNoise_2./noise_2;
actualNoise = randomNoiseRow.*repmat(sqrt(ratio),size(randomNoiseRow,1),1);
actualNoise = reshape(actualNoise,size(randomNoise));

function SNR = calcSNR(origSignal, noisySignal)
errorSignal = origSignal-noisySignal;
signal_2 = sum(origSignal.^2);
noise_2 = sum(errorSignal.^2);

SNRValues = 10*log10(signal_2./noise_2);
SNR = mean(SNRValues);
