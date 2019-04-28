% Applies Generalized Anscombe variance-stabilizing transformation [1].
%
% Syntax:     fz = GenAnscombe_forward(z, sigma, alpha, g)
%
%
% The input signal  z  is assumed to follow the Poisson-Gaussian noise model
%
%    z = alpha * p + n
%
% where  p  is a Poisson-distributed realization of the unknown original
% signal intensity,  alpha  is a positive scaling factor, and  n  is additive
% Gaussian noise with mean  g  and variance  sigma^2.
%
% Only  z  and  sigma  are required inputs.  If not specified, the parameter
% alpha  is assumed to be equal to 1  and  g  is assumed to be equal to 0.
%
%
% The output  fz  has variance approximately equal to 1.
%
%
% References:
% [1] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing  and  Data Analysis, Cambridge University Press, Cambridge, 1998)
%
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2011
% -------------------------------------------------------------------------------

function fz = GenAnscombe_forward(z, sigma, alpha, g)

if 1
    
    if ~exist('g','var')
        g = 0;
    end
    
    if ~exist('alpha','var')
        alpha = 1;
    end
    
    fz = 2/alpha * sqrt(max(0,alpha*z + (3/8)*alpha^2 + sigma^2 - alpha*g));
    
else  % equivalent
    
    % initial variable change
    if ~exist('g','var')
        %  g = 0;
    else
        z = (z-g);
    end
    
    if ~exist('alpha','var')
        %  alpha = 1;
    else
        z = z/alpha;
        sigma = sigma/alpha;
    end
    
    fz = 2 * sqrt(max(0,z + (3/8) + sigma^2));
    
end

