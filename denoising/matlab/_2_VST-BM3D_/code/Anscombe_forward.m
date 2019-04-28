% Applies Anscombe variance-stabilizing transformation
%
%  transformed = Anscombe_forward(z)
%
%  the input  z  is noisy Poisson-distributed data
%
%
%  the output  fz  has variance approximately equal to 1.
%
%
% Reference:
% Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2009
% -------------------------------------------------------------------------------

function transformed = Anscombe_forward(z)

transformed = 2*sqrt(z+3/8);  % Apply Anscombe variance-stabilizing transformation