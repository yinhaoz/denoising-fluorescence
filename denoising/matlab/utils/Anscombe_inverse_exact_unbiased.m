function exact_inverse = Anscombe_inverse_exact_unbiased(D)
% Applies exact unbiased inverse of Anscombe variance-stabilizing transformation
%
%  exact_inverse = Anscombe_inverse_exact_unbiased(D)
%
%  the input  D  is the filtered (e.g., denoised) signal obtained by processing after variance-stabilization with the Anscombe forward transformation.
%
%
%  This function requires the file  Anscombe_vectors.mat
%
%
% References:
% [1] M. Mäkitalo and A. Foi, "On the inversion of the Anscombe transformation in low-count Poisson image denoising", Proc. Int. Workshop on Local and Non-Local Approx. in Image Process., LNLA 2009, Tuusula, Finland, pp. 26-32, August 2009. doi:10.1109/LNLA.2009.5278406
% [2] M. Mäkitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
% [3] Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2009
% -------------------------------------------------------------------------------

load Anscombe_vectors.mat Efz Ez   % load the pre-computed variables Efz and Ez, i.e. E{f(z)|y} and E{z|y}=y, respectively where f(z)=2*sqrt(z+3/8);

asymptotic = (D/2).^2 - 1/8;   % asymptotically unbiased inverse [3]

exact_inverse = interp1(Efz,Ez,D,'linear','extrap');   % exact unbiased inverse [1,2]

outside_exact_inverse_domain = D > max(Efz(:));    % for large values use asymptotically unbiased inverse instead of linear extrapolation of exact unbiased inverse outside of pre-computed domain
exact_inverse(outside_exact_inverse_domain) = asymptotic(outside_exact_inverse_domain);

outside_exact_inverse_domain = D < 2*sqrt(3/8);% min(Efz(:));
exact_inverse(outside_exact_inverse_domain) = 0;
