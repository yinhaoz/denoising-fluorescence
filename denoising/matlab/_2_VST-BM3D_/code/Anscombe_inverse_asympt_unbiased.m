% Applies asyptotically unbiased inverse of Anscombe variance-stabilizing transformation
%
%  asymptotic = Anscombe_inverse_asympt_unbiased(D)
%
%  the input  D  is the filtered (e.g., denoised) signal obtained by processing after variance-stabilization with the Anscombe forward transformation.
%
%
% Reference:
% Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2009
% -------------------------------------------------------------------------------

function asymptotic = Anscombe_inverse_asympt_unbiased(D)

asymptotic = (D/2).^2 - 1/8;   % asymptotically unbiased inverse

asymptotic(D < 2*sqrt(3/8)) = 0;