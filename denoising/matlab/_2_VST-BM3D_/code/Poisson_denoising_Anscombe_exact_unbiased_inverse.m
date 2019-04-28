% Denoise Poisson images using the Anscombe variance-stabilizing transformation, BM3D filter, and the exact unbiased inverse of the Anscombe transformation
%
%  [y_hat, PSNR_y_hat, NMISE_y_hat] = Poisson_denoising_Anscombe_exact_unbiased_inverse(noisy, original)
%
%  the input  D  is the filtered (e.g., denoised) signal obtained by processing after variance-stabilization with the Anscombe forward transformation.
%
%
% References:
% [1] M. Mäkitalo and A. Foi, "On the inversion of the Anscombe transformation in low-count Poisson image denoising", Proc. Int. Workshop on Local and Non-Local Approx. in Image Process., LNLA 2009, Tuusula, Finland, pp. 26-32, August 2009. doi:10.1109/LNLA.2009.5278406
% [2] M. Mäkitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
% [3] F.J. Anscombe, "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2009
% -------------------------------------------------------------------------------

function [y_hat, PSNR_y_hat, NMISE_y_hat] = Poisson_denoising_Anscombe_exact_unbiased_inverse(noisy, original)

%% Apply Anscombe variance-stabilizing transformation
transformed=Anscombe_forward(noisy);  % Apply Anscombe variance-stabilizing transformation [3]
transformed_sigma=1;   %% this is the standard deviation assumed for the transformed data


%% Scale the image (BM3D processes inputs in [0,1] range)  (these are affine transformations)
maxtransformed=max(transformed(:));   %% first put data into [0,1] ...
mintransformed=2*sqrt(0+3/8); % min(transformed(:));
transformed=(transformed-mintransformed)/(maxtransformed-mintransformed);
transformed_sigma=transformed_sigma/(maxtransformed-mintransformed);

scale_range=0.7;  %% ... then set data range in [0.15,0.85], to avoid clipping of extreme values
scale_shift=(1-scale_range)/2;
transformed=transformed*scale_range+scale_shift;
transformed_sigma=transformed_sigma*scale_range;


%% Denoise transformed data (BM3D)
% disp(['Min: ',num2str(min(transformed(:))),'   Max: ',num2str(max(transformed(:))),'   sigma*255: ',num2str(transformed_sigma*255)]);
if exist('BM3D.m','file')
    [dummy D]=BM3D(1,transformed,transformed_sigma*255,'np');  % denoise assuming additive white Gaussian noise
else
    disp(' '),disp(' '),disp(' '),disp(' !!!  BM3D denoising software not found  !!!'),disp(' '),disp('     BM3D can be downloaded from http://www.cs.tut.fi/~foi/GCF-BM3D/ '),disp(' '),disp(' ')
    return
end


%% Invert scaling back to the initial VST range (these are affine transformations)
D=(D-scale_shift)/scale_range;
D=D*(maxtransformed-mintransformed)+mintransformed;


%% Inversion of the variance-stabilizing transformation
y_hat=Anscombe_inverse_exact_unbiased(D);  % apply exact unbiased inverse of the Anscombe variance-stabilizing transformation

% %  The exact unbiased inverse provides superior results than those conventionally obtained using the asymptotically unbiased inverse (D/2).^2 - 1/8
% y_hat=Anscombe_inverse_asympt_unbiased(D);  % apply asymptotically unbiased inverse of the Anscombe variance-stabilizing transformation



%% If the original image is provided, we compute the PSNR and NMISE values
if nargin == 2
    peak = max(original(:));
    PSNR_y_hat = 10*log10(peak^2/mean((y_hat(:)-original(:)).^2));
    
    % Remove possible zeroes from the original image, and then the corresponding
    % values from the estimate y_hat (to avoid problems in computing the NMISE)
    vec_original = original(:);
    zeroes = find(vec_original == 0);
    vec_original(zeroes) = [];
    
    vec_y_hat = y_hat(:);
    vec_y_hat(zeroes) = [];
    
    % Normalized mean integrated square error of the estimate y_hat
    N = numel(vec_original);
    NMISE_y_hat = sum((vec_y_hat-vec_original).^2 ./ vec_original) / N;
else
    PSNR_y_hat = -1;
    NMISE_y_hat = -1;
end



%% end of code



