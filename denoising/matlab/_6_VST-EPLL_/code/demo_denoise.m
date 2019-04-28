clear
patchSize = 8;

% load image
I = double(rgb2gray(imread('160068.jpg')))/255;	

% add noise
noiseSD = 25/255;
noiseI = I + noiseSD*randn(size(I));
excludeList = [];

% set up prior
LogLFunc = [];
load GSModel_8x8_200_2M_noDC_zeromean.mat
prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);

%%
tic
% add 64 and 128 for high noise
[cleanI,psnr,~] = EPLLhalfQuadraticSplit(noiseI,patchSize^2/noiseSD^2,patchSize,(1/noiseSD^2)*[1 4 8 16 32],1,prior,I,LogLFunc);
toc

% output result
figure(1);
imshow(I); title('Original');
figure(2);
imshow(noiseI); title('Corrupted Image');
figure(3);
imshow(cleanI); title('Restored Image');
fprintf('PSNR is:%f\n',20*log10(1/std2(cleanI-I)));


