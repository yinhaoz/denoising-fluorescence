%DENOISING_DEMO: Denoising demonstration based on the PURE-LET principle
%   applied to redundant transform-domain thresholding.
%
%   See also UWT_PURELET_denoise, BDCT_PURELET_denoise and 
%   DICT_SURELET_denoise.
%
%   Author:
%   Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%
%   References:
%   [1] F. Luisier, T. Blu, M. Unser, "Image Denoising in Mixed Poisson-
%   Gaussian Noise," IEEE Transactions on Image Processing, vol. 20, no 3,
%   March 2011.
%   [2] F. Luisier, "The SURE-LET Approach to Image Denoising," Swiss
%   Federal Institute of Technology Lausanne, EPFL Thesis no. 4566 (2010),
%   232 p., January 8, 2010.

clearvars;
restoredefaultpath;
path(path,'./Denoising');
path(path,'./diagDR');
path(path,'./Images');
path(path,'./Transforms');
path(path,'./Utilities');

% If DISPLAY = 'on', the original, noisy and denoised images are displayed
DISPLAY   = 'on';
% 'UWT' or 'BDCT' or 'DICT'
transform = 'uwt'; 
% Size of the square DCT blocks (8, 12 or 16)
params.bsize = 16;
% Type of wavelet filter (this version only works with Haar filters)
params.wtype = 'haar';
% If boundary = 1 (default), mirror boundary conditions are applied
% If boundary = 0, periodic boundary conditions are applied
params.boundary = 1; 
% Number of BDCT radial frequency bands (default is 6)
params.nband = 6;

% Load the original noise-free image
%-----------------------------------
filename = 'cameraman.tif';
original = double(imread(filename));

% Rescale the original noise-free image
%--------------------------------------
Imax     = 60;
sigma    = 1*Imax/10;
original = Imax*original/max(original(:)); 

% Create input noisy image
%-------------------------
RandStream.setGlobalStream(RandStream('mt19937ar','seed',0));
wgn   = randn(size(original));
wgn   = sigma*wgn/std2(wgn);
input = poissrnd(original)+wgn;

% Denoise
%--------
start = clock;
switch lower(transform)
    case 'uwt'
        output = UWT_PURELET_denoise(input,sigma,params);
    case 'bdct'
        output = BDCT_PURELET_denoise(input,sigma,params);
    case 'dict'
        output = DICT_PURELET_denoise(input,sigma,params);
    otherwise
        disp('Unknown transform !');
        return;
end
time = etime(clock,start);       

% PSNR computation
%-----------------
MSE_0  = mean((input(:)-original(:)).^2);
MSE_D  = mean((output(:)-original(:)).^2);
PSNR_0 = 10*log10(Imax^2/MSE_0);
PSNR_D = 10*log10(Imax^2/MSE_D);

% Display PSNR results
%---------------------
fprintf(['\nInput PSNR   : ' num2str(PSNR_0,'%.2f') '[dB]']);
fprintf(['\nOutput PSNR  : ' num2str(PSNR_D,'%.2f') '[dB]']);
fprintf(['\nElapsed time : ' num2str(time,'%.2f') '[s]\n\n']);

% Display results
%----------------
if(strcmp(DISPLAY,'on'))
    h = figure('Units','normalized','Position',[0 0.3 1 0.5]);
    set(h,'name','PURE-LET Denoising');
    subplot(1,3,1);imagesc(uint8(255*original/Imax),[0 255]);
    axis image;colormap gray(256);axis off;
    title('Original','fontsize',16,'fontweight','bold');drawnow;
    subplot(1,3,2);imagesc(uint8(255*input/Imax),[0 255]);
    axis image;colormap gray(256);axis off;
    title(['Noisy: PSNR = ' num2str(PSNR_0,'%.2f') ' dB'],...
        'fontsize',16,'fontweight','bold');drawnow;
    subplot(1,3,3);imagesc(uint8(255*output/Imax),[0 255]);
    axis image;colormap gray(256);axis off;
    title(['Denoised: PSNR = ' num2str(PSNR_D,'%.2f') ' dB'],...
        'fontsize',16,'fontweight','bold');drawnow;
end