% Applies closed-form approximation of the exact unbiased inverse of Generalized Anscombe variance-stabilizing transformation
%
%
% Syntax:
%
%  exact_inverse = GenAnscombe_inverse_closed_form(D,sigma,alpha,g)
%
%
% Inputs: 
%
%    D  is the filtered (e.g., denoised) signal obtained by processing after variance-stabilization with the Generalized Anscombe forward transformation.
%   sigma  is the standard-deviation of the Gaussian noise component in the observations (i.e. before stabilization of variance).
%   alpha  is the positive scaling factor of the Poisson component in the observations.
%    g  is the mean of the Gaussian noise component in the observations.
%
%  Only  D  and  sigma  are required inputs.  If not specified, the parameter
%  alpha  is assumed to be equal to 1  and  g  is assumed to be equal to 0.
%
%
% References:
% [1] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", IEEE Trans. Image Process., doi:10.1109/TIP.2012.2202675
% [2] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing  and  Data Analysis, Cambridge University Press, Cambridge, 1998)
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2011-2012
% -----------------------------------------------------------------------------------

function exact_inverse = GenAnscombe_inverse_closed_form(D,sigma,alpha,g)

if ~exist('alpha','var')
    %     alpha = 1;
else
    sigma = sigma/alpha;
end

exact_inverse = (D/2).^2 + 1/4*sqrt(3/2)*D.^-1 - 11/8*D.^-2 + 5/8*sqrt(3/2)*D.^-3 - 1/8 - sigma.^2; % closed-form approximation of the exact unbiased inverse
exact_inverse = max(0,exact_inverse);


%% reverse the initial variable change

if ~exist('alpha','var')
    %     alpha = 1;
else
    exact_inverse=exact_inverse*alpha;
end

if ~exist('g','var')
    %     g = 0;
else
    exact_inverse=exact_inverse+g;
end

