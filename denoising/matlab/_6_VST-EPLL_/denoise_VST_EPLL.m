function [img_denoise, time] = denoise_VST_EPLL(img_raw)

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

    %%  apply forward variance stabilizing transformation
    fz = GenAnscombe_forward(img_raw, sigma, a, 0);
    sigma_den = 1; % Standard-deviation value assumed after variance-stabiliation
    
    % scale the transformed image to [0, 1]
    scale_range = 1;
    scale_shift = (1-scale_range)/2;
    maxzans = max(fz(:));
    minzans = min(fz(:));
    fz = (fz-minzans)/(maxzans-minzans);   sigma_den = sigma_den/(maxzans-minzans);
    fz = fz*scale_range+scale_shift;      sigma_den = sigma_den*scale_range;
    
    %% perform denoising algorithm on transformed images
    noiseSD = sigma_den;
    patchSize = 8;
    excludeList = [];
    LogLFunc = [];
    load GSModel_8x8_200_2M_noDC_zeromean.mat
    prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);

    [D,~,~] = EPLLhalfQuadraticSplit(fz,...
        patchSize^2/noiseSD^2, patchSize, (1/noiseSD^2)*[1 4 8 16 32],...
        1, prior, zeros(size(fz)), LogLFunc);

    % Scale back to the initial VST range
    D = (D-scale_shift)/scale_range;
    D = D*(maxzans-minzans)+minzans;
    
    %%  apply the inverse VST transform
    img_denoise = GenAnscombe_inverse_exact_unbiased(D, sigma, a, 0);   % exact unbiased inverse

    time = etime(clock, t0);
    
    fprintf('Denoising operation ends\n');
end