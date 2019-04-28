function [img_denoise, time] = denoise_PURE_LET(img_raw)

    fprintf('Denoising operation starts\n');
    
    %% estimate the noise of raw image
    fitparams = estimate_noise(img_raw);
    a = fitparams(1);
    b = fitparams(2);
    if a<0
        a = eps;
    end
    if b<0
        b = eps;
    end
    sigma = sqrt(b);
    
    t0=clock;  
    
    %% perform denoising algorithm
    
    input = img_raw/a; % unit: # photons
    sigma = sigma/a;
    
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
    img_denoise = output*a;
    
    time = etime(clock, t0);
    
    fprintf('Denoising operation ends\n');
end