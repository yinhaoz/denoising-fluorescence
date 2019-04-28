% Applies exact unbiased inverse of Generalized Anscombe variance-stabilizing transformation
%
%
% Syntax:
%
%  exact_inverse = GenAnscombe_inverse_exact_unbiased(D,sigma,alpha,g)
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
%  This function requires the files  GenAnscombe_vectors.mat
%                                    Anscombe_vectors.mat
%                                    Anscombe_inverse_exact_unbiased.m
%
%
% References:
% [1] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", IEEE Trans. Image Process., doi:10.1109/TIP.2012.2202675
% [2] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing  and  Data Analysis, Cambridge University Press, Cambridge, 1998)
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2011-2012
% -----------------------------------------------------------------------------------

function exact_inverse = GenAnscombe_inverse_exact_unbiased(D,sigma,alpha,g)

load GenAnscombe_vectors.mat  %  Efzmatrix Ezmatrix sigmas

if ~exist('alpha','var')
    %     alpha = 1;
else
    sigma = sigma/alpha;
end

%% interpolate the exact unbiased inverse for the desired sigma
% sigma is given as input parameter
if (sigma > max(sigmas))
    % for very large sigmas, use the exact unbiased inverse of Anscombe modified by a -sigma^2 addend
    exact_inverse = Anscombe_inverse_exact_unbiased(D) - sigma^2;
    exact_inverse = max(0,exact_inverse);  %% this should be necessary, since Anscombe_inverse_exact_unbiased(D) is >=0 and sigma>=0.
elseif sigma > 0
    % interpolate Efz
    Efz = interp2(sigmas,Ez,Efzmatrix,sigma,Ez);
    
    % apply the exact unbiased inverse
    exact_inverse = interp1(Efz,Ez,D,'linear','extrap');
    
    % outside the pre-computed domain, use the exact unbiased inverse of Anscombe modified by a -sigma^2 addend
    % (the exact unbiased inverse of Anscombe takes care of asymptotics)
    outside_exact_inverse_domain = D > max(Efz(:));
    asymptotic = Anscombe_inverse_exact_unbiased(D) - sigma^2;
    exact_inverse(outside_exact_inverse_domain) = asymptotic(outside_exact_inverse_domain);
    outside_exact_inverse_domain = D < min(Efz(:));
    exact_inverse(outside_exact_inverse_domain) = 0;
elseif sigma == 0
    % if sigma is zero, then use exact unbiased inverse of Anscombe
    % transformation (higher numerical precision)
    exact_inverse = Anscombe_inverse_exact_unbiased(D);
else % sigma < 0
    error(' Error: sigma must be non-negative! ')
    return
end


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
