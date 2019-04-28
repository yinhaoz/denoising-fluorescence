% Applies MMSE inverse of Anscombe variance-stabilizing transformation
%
%  I_MMSE = Anscombe_inverse_exact_unbiased(D)
%
%  Inputs:
%     D   is the filtered (e.g., denoised) signal obtained by processing after variance-stabilization with the Anscombe forward transformation.
%   stdD  is the standard deviation of D, assuming D~N(E{f(z)|y},stdD^2).
%
%
%  This function requires the files  Anscombe_inverse_exact_unbiased.m, Anscombe_vectors.mat, and MMSEcurves.mat
%
%
% References:
%  [1] M. Mäkitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
%  [2] Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2009
% -------------------------------------------------------------------------------

function I_MMSE = Anscombe_inverse_MMSE(D,stdD)

load MMSEcurves.mat

sigma_index_matrix=max(1,min(numel(sigmas),interp1(sigmas,1:numel(sigmas),stdD,'linear','extrap')));  %% max(min( just in case
D_index_matrix=interp1(D1_values,1:numel(D1_values),D,'linear','extrap');  %% here we can have problems when D is too large or too small

%% D can be either too large or too small with respect to the precomputed values in D1_values
% D is too large
clip_domain=D_index_matrix>numel(D1_values);
D_index_matrix_clipped=D_index_matrix;
D_index_matrix_clipped(clip_domain)=numel(D1_values);
I_MMSE=interp2(y_hats_values,sigma_index_matrix,D_index_matrix_clipped);
I_MMSE(clip_domain)=I_MMSE(clip_domain)-Anscombe_inverse_exact_unbiased(max(D1_values))+Anscombe_inverse_exact_unbiased(D(clip_domain));

% D is too small (this should not happen, because in D1_values there is also the value 0)
clip_domain=D_index_matrix<1;
D_index_matrix_clipped=D_index_matrix;
D_index_matrix_clipped(clip_domain)=1;
% if 0
%     I_MMSE(clip_domain)=interp2(y_hats_values,sigma_index_matrix(clip_domain),D_index_matrix_clipped(clip_domain)).*(1-D_index_matrix(clip_domain)+1)+interp2(y_hats_values,sigma_index_matrix(clip_domain),D_index_matrix_clipped(clip_domain)+1).*(D_index_matrix(clip_domain)-1);  %% linear extrapolation (better than nothing)
%     I_MMSE(clip_domain)=max(0,I_MMSE(clip_domain));  %% linear extrapolation can produce negative values (which is nonsense)
% else
I_MMSE(clip_domain)=exp(log(interp2(y_hats_values,sigma_index_matrix(clip_domain),D_index_matrix_clipped(clip_domain))).*(1-D_index_matrix(clip_domain)+1)+log(interp2(y_hats_values,sigma_index_matrix(clip_domain),D_index_matrix_clipped(clip_domain)+1)).*(D_index_matrix(clip_domain)-1));  %% extrapolation
% end
