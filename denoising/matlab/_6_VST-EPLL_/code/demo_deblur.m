clear
patchSize = 8;

I = double(rgb2gray(imread('160068.jpg')))/255;

% load blurring kernel (you can download the kernels from Dilip Krishnan's
% website
% load kernels.mat
% K = kernel1;
K = fspecial('motion',10,45);
% K = fspecial('gaussian',[5 5],1);
noiseSD = 0.01;
patchSize = 8;


% convolve with kernel and add noise
ks = floor((size(K, 1) - 1)/2);
yorig = I;
y = conv2(yorig, K, 'valid');
y = y + noiseSD*randn(size(y));
y = double(uint8(y .* 255))./255;

% code excerpt taken from Krishnan et al.

% edgetaper to better handle circular boundary conditions
y = padarray(y, [1 1]*ks, 'replicate', 'both');
for a=1:4
  y = edgetaper(y, K);
end

noiseI = y;

% load GMM model
load GSModel_8x8_200_2M_noDC_zeromean.mat

% uncomment this line if you want the total cost calculated
% LogLFunc = @(Z) GMMLogL(Z,GS); 

% initialize prior function handle
excludeList = [];
prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);

% comment this line if you want the total cost calculated
LogLFunc = [];

% deblur
tic
[cleanI,psnr,~] = EPLLhalfQuadraticSplitDeblur(noiseI,64/noiseSD^2,K,patchSize,50*[1 2 4 8 16 32 64],1,prior,I,LogLFunc);
toc

% output result
figure(1);
imshow(I); title('Original');
figure(2);
imshow(noiseI); title('Corrupted Image');
figure(3);
imshow(cleanI); title('Restored Image');