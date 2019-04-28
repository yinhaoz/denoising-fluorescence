% Applies the closed-form approximation of the exact unbiased inverse of Anscombe variance-stabilizing transformation
%
%  exact_inverse = Anscombe_inverse_closed_form(D)
%
%  the input  D  is the filtered (e.g., denoised) signal obtained by processing after variance-stabilization with the Anscombe forward transformation.
%
%
%  This function requires the file  Anscombe_vectors.mat
%
%
% References:
% [1] M. Mäkitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
% [2] M. Mäkitalo and A. Foi, "A closed-form approximation of the exact unbiased inverse of the Anscombe variance-stabilizing transformation", IEEE Trans. Image Process., vol. 20, no. 9, pp. 2697-2698, September 2011. doi:10.1109/TIP.2011.2121085
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2010
% -------------------------------------------------------------------------------

function exact_inverse = Anscombe_inverse_closed_form(D)

exact_inverse = (D/2).^2 + 1/4*sqrt(3/2)*D.^-1 - 11/8*D.^-2 + 5/8*sqrt(3/2)*D.^-3 - 1/8; % closed-form approximation of the exact unbiased inverse
exact_inverse = max(0,exact_inverse);
