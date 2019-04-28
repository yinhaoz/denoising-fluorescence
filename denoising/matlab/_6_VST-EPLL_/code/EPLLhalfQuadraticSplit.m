function [cleanI,psnr,cost] = EPLLhalfQuadraticSplit(noiseI,lambda,patchSize,betas,T,prior,I,LogLFunc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% EPLLhalfQuadraticSplit - Minimizes the EPLL cost using half quadratic splitting
% as defined in the paper:
% "From Learning Models of Natural Image Patches to Whole Image Restoration"
% by Daniel Zoran and Yair Weiss, ICCV 2011
% 
% 
% Version 1.0 (21/10/2011)
%
% This function is for denoising and inpainting - for deblurring refer to
% EPLLhalfQuadraticSplitDeblur.m
%
% Inputs:
%
%   noiseI - the noisy image
%   lambda - the parameter lambda from Equation (2) in the paper (mostly
%            used as the inverse of the noise variance. If a matrix is given, it
%            should be the same size as the image (used for inpainting)
%   patchSize - the size of patches to extract (single scalar, patches are
%               always square)
%   betas - a list (1xM vector) of beta values, if the values are positive, they will be
%           used as is, negative values will be ignored and beta will be estimated
%           automatically from the noisy image (for as many iterations as there are
%           in betas)
%   T - The number of iterations to optimizie for X and Z at each beta
%       value
%   prior - a function handle to a function which calculates a MAP estimate
%           using a given prior for a noisy patch at noise level beta, see examples in the
%           demos
%   I - the original image I, used only for PSNR calculations and
%       comparisons
%   LogLFunc - a function handle to calculate the log likelihood of patches
%              in the image, used for calculating the total cost (optional).
%
%
%
% Outputs:
%
%   cleanI - the restored image
%   psnr - a list of the psnr values obtained for each beta and iteration
%   cost - if LogLFunc is given then this is the cost from Equation 2 in
%          the paper at each value of beta.
%
% See demos in this same code package for examples on how to use this
% function for denoising and inpainting using some example priors
% (including the GMM prior used in the paper).
%
% All rights reserved to the authors of the paper (Daniel Zoran and Yair
% Weiss). If you have any questions, comments or suggestions please contact
% Daniel Zoran at daniez@cs.huji.ac.il.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% estimate the "real" noise standard deviation from lambda
RealNoiseSD = sqrt(1/(lambda(1)/patchSize^2));

% See if the user expects a cost and if the LogLFunc function handle was
% given
calc_cost = false;
cost = [];
if nargout>2
    if isa(LogLFunc,'function_handle')
        calc_cost = true;
    else
        calc_cost = false;
    end
end

% Just a simple guess for beta in case the auto-estimation doesn't work for
% some reason 
beta = abs(betas(1)/4);


% initialize with the noisy image
cleanI = noiseI;

k=1;
sd = Inf;

% go through all values of beta
for betaa=betas
    
    % if any beta<0 estimate betas automatically
    if (betaa<0)
        old_sd = sd;
        [sd] = estimateNoiseSDUsingKurts(cleanI,12);
        fprintf('sd is:%f beta is %f * (1/noiseSD^2)',sd,(1/sd^2)/(1/RealNoiseSD^2));
        
        % if estimation failed or doesn't make sense, just guess, rarely
        % used
        if isnan(sd) || sd>old_sd
            beta = beta*4;
            sd = beta^-0.5;
        else
            beta = 1/sd^2;
        end
    else
        beta = betaa;
    end
    
    % go for T iterations, optimizing for X and Z at each iteration
    for tt=1:T
        
        % Z step
        
        % Extract Z, all overlapping patches from the current estimate
        Z = im2col(cleanI,[patchSize patchSize]);

        % calculate orignal cost if LogLFunc is defined and output
        % arguments expect it
        if calc_cost
            cost(k) = 0.5*lambda*sum((cleanI(:)-noiseI(:)).^2) - EPLL(Z,LogLFunc);
            fprintf('Cost is: %f\n',cost(k));
        end
        
        % calculate the MAP estimate for Z using the given prior
        cleanZ = prior(Z,patchSize,(beta)^-0.5,size(noiseI));
        
        % X step
        
        % average the pixels in the cleaned patches in Z
        [I1] = scol2im(cleanZ,patchSize,size(I,1),size(I,2),'average');
        counts = patchSize^2;
        
        % and calculate the current estimate for the clean image
        cleanI = noiseI.*lambda./(lambda+beta*counts) + (beta*counts./(lambda+beta*counts)).*I1;
                
        % calculate the PSNR for this step
        psnr(k) = 20*log10(1/std2(cleanI-I));
        
        % output the result to the console
        fprintf('PSNR is:%f I1 PSNR:%f\n',psnr(k),20*log10(1/std2(I1-I)));
%       imshow([I noiseI cleanI],[0 1]); drawnow;

        k=k+1;      
    end
end

% output the clean estimate
cleanI = reshape(cleanI,size(noiseI));

% clip values to be between 1 and 0, hardly changes performance
cleanI(cleanI>1)=1;
cleanI(cleanI<0)=0;
psnr = psnr(:)';