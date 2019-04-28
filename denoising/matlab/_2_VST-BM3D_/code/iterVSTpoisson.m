function yhat=fun_IterPoissonBM3D(z)
% Iterative Poisson denoising algorithm based on variance stabilization and AWGN denoising (v1.00)
%
% INPUT:   z   Poisson noisy image   z ~ P(y)
% OUTPUT: yhat  denoised image,  estimate of y
%
%
% L. Azzari and A. Foi, "Variance Stabilization for Noisy+Estimate
% Combination in Iterative Poisson Denoising", submitted, March 2016
%
% http://www.cs.tut.fi/~foi/invansc/
%
%  L. Azzari and Alessandro Foi - Tampere University of Technology - 2016 - All rights reserved.
% -----------------------------------------------------------------------------------------------
%

%% Prepare functions
load Anscombe_lambda lambdaGridTimesE yGrid lambdaGrid   % load expectation precomputed on a grid for interpolation of the exact unbiased inverse

Ediff = @(D,lambda) ((lambda-1)/lambda)./(D.^3);  % eq. (9) (difference of expectations)
% asymInv=@(x) (x.^2/4-1/8);                      % asymptotically unbiased inverse of Anscombe (coincides with exact unbiased for large y)
f=@(x,lambda) 2*sqrt(max(0,x/lambda^2+3/8));      % forward VST (Anscombe of scaled input)


%% Parameter setup

load paramsFromQfun   % provide algorithm parameters based on histogram (quantiles) of noisy image.
Qs=linspace(0,1,numel(whichQs));     % quantiles of interest  (typically first and last, i.e. min and max are ignored, as they are unstable)
QQ=quantile(z(:),Qs(whichQs));       % vector of quantiles of z
paramsFromQ=paramsFromQfun(Ps(QQ));
lambda_K=paramsFromQ(1);
K=paramsFromQ(2);
h_1=1+2*paramsFromQ(3);
h_K=1+2*paramsFromQ(4);
lambda_1=1;

if K>1
    lambdaS=linspace(lambda_1,lambda_K,K);    % lambdas
else
    lambdaS=lambda_1;
    lambda_K=lambda_1;
end

hS=max(h_K,h_1-2*(1:K)+2);     % bin sizes

% remove steps with too small lambda, and update end values
hS(lambdaS<0.01)=[];
lambdaS(lambdaS<0.01)=[];
lambda_K=lambdaS(end);
% h_K=hS(end);

if lambda_K>0.99 % if last lambda is very close to 1, then do only last iteration, because previous ones have negligible effect in the convex combination
    hS=h_K;
    lambdaS=1;
    % lambda_K=1;
end

% print out lambda and h parameters to be used during the iterations
disp([['  + lambda:  ';'  | h:       '],num2str([lambdaS;hS])])



%% Main loop of Algorithm 1
timeOld=now;

yhat=z;
for i=1:numel(lambdaS)
    
    lambda=lambdaS(i);  % lambda for current iteration
    h=hS(i);            % bin size for current iteration
    
    if lambda>0;  % if lambda=0, there is no noise in z_i, thus previous estimate of yhat is not modified
        
        %% convex combination
        
        if i>1
            z_i=lambda*z+(1-lambda)*yhat;
        else
            z_i=z;
        end
   
        
        %% Binning
        z_B=bin_B_h(z_i,h);
        
        %% Apply forward VST
        fz = f(z_B,lambda);
        
        
        %% AWGN DENOISING
        
        sigma_den = 1;  % Standard-deviation value assumed after variance stabiliation
        
        % Scale the image (BM3D processes inputs in [0,1] range)
        scale_range = .9;
        scale_shift = (1-scale_range)/2;
        maxzans = max(fz(:));
        minzans = min(fz(:));
        fz   =   (fz-minzans)/(maxzans-minzans);
        sigma_den = sigma_den/(maxzans-minzans);
        fz = scale_shift + fz*scale_range;
        sigma_den = sigma_den*scale_range;
        
        [dummy D] = BM3D(0,fz,sigma_den*255,'np',0); % denoise assuming AWGN using BM3D http://www.cs.tut.fi/~foi/GCF-BM3D/
        
        % Scale back to the initial VST range
        D = (D-scale_shift)/scale_range;
        D = D*(maxzans-minzans)+minzans;
        
        
        %% Apply the inverse VST for convex combination z_i of Poisson z and estimate yhat
        
        if lambda==1
            
            yhat = Anscombe_inverse_exact_unbiased(D);   % exact unbiased inverse of pure Poisson  http://www.cs.tut.fi/~foi/invansc/
            
        else
            
            yhat = zeros(size(D));
            if lambda<min(lambdaGrid)
                E = f(yGrid,lambda);   % if lambda is very small, then z_bar is almost deterministic, hence E of f is like f of E
            else
                E = interp2(yGrid,lambdaGrid',lambdaGridTimesE,yGrid,lambda,'linear',0)'/lambda;
            end
            
            III = (D<=max(E(:)));
            yhat(III)    = interp1(E,yGrid',D(III), 'linear', 'extrap');
            yhat(~III)   = Anscombe_inverse_exact_unbiased(D(~III)+Ediff(D(~III),lambda))*lambda^2;
            % yhat(~III) = asymInv(D(~III)+Ediff(D(~III),lambda))*lambda^2; % alternative to using Anscombe_inverse_exact_unbiased
            
            III = (D<min(E(:)));
            yhat(III)    = 0;
        end
        
        
        %% Debinning
        J=9;
        yhat=debin_Binv_h(yhat,size(z),h,J);
        
    end
    
    %% end of loop
end
%% end of Algorithm 1

timeNow=now;
disp(['  + Elapsed ', num2str(86400*(timeNow-timeOld)),'s']);
