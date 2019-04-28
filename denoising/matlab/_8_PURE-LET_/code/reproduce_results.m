%REPRODUCE_RESULTS: Reproduce the results presented in [1].
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

% Noise model
%------------
Imax = [
    120;
    60;
    30;
    20;
    10;
    5;
    2;
    1;
];
NI = numel(Imax);
% Standard deviation of the AWGN
sigma = 0*Imax/10;
% Number of noise realizations
NR = 10;

PSNRin  = zeros(NI,NR);
PSNRout = zeros(NI,NR);
Time    = zeros(NI,NR);
fprintf('\n');
for I = 1:NI
    % Rescale the original noise-free image
    %--------------------------------------
    original = Imax(I)*original/max(max(original));
    for R = 1:NR
        fprintf('o');
        % Create input noisy image
        %-------------------------
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',R-1));
        wgn   = randn(size(original));
        wgn   = sigma(I)*wgn/std2(wgn);
        input = poissrnd(original)+wgn;
        % Denoise
        %--------
        start = clock;
        switch lower(transform)
            case 'uwt'
                output = UWT_PURELET_denoise(input,sigma(I),params);
            case 'bdct'
                output = BDCT_PURELET_denoise(input,sigma(I),params);
            case 'dict'
                output = DICT_PURELET_denoise(input,sigma(I),params);
            otherwise
                disp('Unknown transform !');
                return;
        end
        Time(I,R) = etime(clock,start);
        % PSNR computation
        %-----------------
        MSEin  = mean((input(:)-original(:)).^2);
        MSEout = mean((output(:)-original(:)).^2);
        PSNRin(I,R)  = 10*log10(Imax(I)^2/MSEin);
        PSNRout(I,R) = 10*log10(Imax(I)^2/MSEout);
    end
end
fprintf('\n');
avgPSNRin  = mean(PSNRin,2)
avgPSNRout = mean(PSNRout,2)
avgTime    = mean(Time,2)