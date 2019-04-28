clear
patchSize = 8;

% images courtesy of Stefan Roth
I = double((imread('new.jpg')))/255;
mask = double((imread('new_mask.png')))/255;

if size(I,3)>1
    I = rgb2ycbcr(I);
end

% find which patches are occluded for faster performance
noiseI = I;
mask_inds = find(mask>0);
tt = noiseI(:,:,1);
tt(mask_inds)=NaN;
ttt = im2col(tt,[patchSize patchSize]);
excludeList = find(any(isnan(ttt)));
clear ttt;

noiseI = I;
for i=1:size(I,3)
    tt = I(:,:,i);
    tt(mask_inds) = 0;
    noiseI(:,:,i)=tt;
end


% load ICA model and initialize MAP estimator
load ICAModel
W = W*E;
invW = pinv(W);
prior = @(Z,patchSize,noiseSD,imsize) PatchDCTGG(Z,patchSize,noiseSD,imsize,W,invW,excludeList);


%%
tic

% inpainting
lambda = 1000000*ones([size(I,1) size(I,2)]);
lambda(mask_inds)=0;
cleanI = zeros(size(I));
for i=1:size(I,3)
    cleanI(:,:,i) = EPLLhalfQuadraticSplit(noiseI(:,:,i),lambda,patchSize,10*[1 2 16 128 512],20,prior,I(:,:,i));
end

%%
if (size(I,3)>1)
    cleanI = ycbcr2rgb(cleanI);
    I = ycbcr2rgb(I);
    noiseI = ycbcr2rgb(noiseI);
end
toc

%% output result
figure(1);
imshow(I); title('Original');
figure(2);
imshow(noiseI); title('Corrupted Image');
figure(3);
imshow(cleanI); title('Restored Image');
fprintf('PSNR is:%f\n',20*log10(1/std2(cleanI-I)));

