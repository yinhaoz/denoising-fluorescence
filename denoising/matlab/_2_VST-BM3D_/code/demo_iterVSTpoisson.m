% Demo script reproducing the experiments reported in
% L. Azzari and A. Foi, "Variance Stabilization for Noisy+Estimate
% Combination in Iterative Poisson Denoising", submitted, March 2016
%
%  v1.00
%
% http://www.cs.tut.fi/~foi/invansc/
%
%  L. Azzari and Alessandro Foi - Tampere University of Technology - 2016 - All rights reserved.
% -----------------------------------------------------------------------------------------------
%
%   !! IMPORTANT !!
%   This demo requires:
%    * Exact unbiased inverse package "invansc"
%       http://www.cs.tut.fi/~foi/invansc/
%    * BM3D denoising filter
%       http://www.cs.tut.fi/~foi/GCF-BM3D/
% -----------------------------------------------------------------------------------------------

clear all

% y=im2double(imread('.\images_for_table_1\boats512.png'));
y=im2double(imread('.\images_for_table_1\bridge256.png'));
% y=im2double(imread('.\images_for_table_1\cam256.png'));
% y=im2double(imread('.\images_for_table_1\couple512.png'));
% y=im2double(imread('.\images_for_table_1\flag256.png'));
% y=im2double(imread('.\images_for_table_1\hill512.png'));
% y=im2double(imread('.\images_for_table_1\house256.png'));
% y=im2double(imread('.\images_for_table_1\man512.png'));
% y=im2double(imread('.\images_for_table_1\peppers256.png'));
% y=im2double(imread('.\images_for_table_1\saturn256.png'));

% y=im2double(imread('.\images_for_table_2\boats256.png'));
% y=im2double(imread('.\images_for_table_2\bridge256.png'));
% y=im2double(imread('.\images_for_table_2\couple256.png'));
% y=im2double(imread('.\images_for_table_2\hill256.png'));
% y=im2double(imread('.\images_for_table_2\man256.png'));
% y=im2double(imread('.\images_for_table_2\mandrill256.png'));
% y=im2double(imread('.\images_for_table_2\peppers256.png'));


peak=1;

y=y/max(y(:))*peak;

N_reps=1;  % number of noise realizations

repCount=0;
for kk=1:N_reps
    repCount=repCount+1;
    randn('seed',kk);rand('seed',kk);
    z=poissrnd(y);
    yhat=iterVSTpoisson(z);
    PSNR_yhat(repCount) = 10*log10(peak^2/mean((y(:)-yhat(:)).^2));
    disp(['[',num2str(repCount),'/',num2str(N_reps),'] PSNR: ',num2str(PSNR_yhat(repCount))]);
end
disp(['  Av. PSNR: ',num2str(mean(PSNR_yhat))]);

%%
figure
subplot(1,3,1);
imshow(y,[0 peak]),title(['y   peak = ',num2str(peak)])
subplot(1,3,2);
imshow(z,[]),title(['z'])
subplot(1,3,3);
imshow(yhat,[0 peak]),title(['yhat  PSNR: ',num2str(PSNR_yhat(end))])

