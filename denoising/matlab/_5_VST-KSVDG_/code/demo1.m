% KSVD running file
% in this file a synthetic test of the K-SVD algorithm is performed. First,
% a random dictionary with normalized columns is being generated, and then
% a set of data signals, each as a linear combination of 3 dictionary
% element is created, with noise level of 20SNR. this set is given as input
% to the K-SVD algorithm.

% a different mode for activating the K-SVD algorithm is until a fixed
% error is reached in the Sparse coding stage, instead until a fixed number of coefficients is found
% (it was used by us for the
% denoising experiments). in order to switch between those two modes just
% change the param.errorFlag (0 - for fixed number of coefficients, 1 -
% until a certain error is reached).



param.L = 3;   % number of elements in each linear combination.
param.K = 50; % number of dictionary elements
param.numIteration = 50; % number of iteration to execute the K-SVD algorithm.

param.errorFlag = 0; % decompose signals until a certain error is reached. do not use fix number of coefficients.
%param.errorGoal = sigma;
param.preserveDCAtom = 0;

%%%%%%% creating the data to train on %%%%%%%%
N = 1500; % number of signals to generate
n = 20;   % dimension of each data
SNRdB = 20; % level of noise to be added
[param.TrueDictionary, D, x] = gererateSyntheticDictionaryAndData(N, param.L, n, param.K, SNRdB);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% initial dictionary: Dictionary elements %%%%%%%%
param.InitializationMethod =  'DataElements';

param.displayProgress = 1;
disp('Starting to  train the dictionary');

[Dictionary,output]  = KSVD(D,param);

disp(['The KSVD algorithm retrived ',num2str(output.ratio(end)),' atoms from the original dictionary']);

[Dictionary,output]  = MOD(D,param);

disp(['The MOD algorithm retrived ',num2str(output.ratio(end)),' atoms from the original dictionary']);
