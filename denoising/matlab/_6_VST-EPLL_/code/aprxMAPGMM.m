function [Xhat] = aprxMAPGMM(Y,patchSize,noiseSD,imsize,GS,excludeList,SigmaNoise)
% approximate GMM MAP estimation - a single iteration of the "hard version"
% EM MAP procedure (see paper for a reference)
%
% Inputs:
%   Y - the noisy patches (in columns)
%   noiseSD - noise standard deviation
%   imsize - size of the original image (not used in this case, but may be
%   used for non local priors)
%   GS - the gaussian mixture model structure
%   excludeList - used only for inpainting, misleading name - it's a list
%   of patch indices to use for estimation, the rest are just ignored
%   SigmaNoise - if the noise is non-white, this is the noise covariance
%   matrix
%
% Outputs:
%   Xhat - the restore patches


% handle exclusion list - used for inpainting
if ~exist('excludeList','var')
    excludeList = [];
end

% Supports general noise covariance matrices
if (~exist('SigmaNoise','var'))
    SigmaNoise = noiseSD^2*eye(patchSize^2);
end

if ~isempty(excludeList)
    T = Y;
    Y = Y(:,excludeList);
end

% remove DC component
meanY = mean(Y);
Y = bsxfun(@minus,Y,meanY);

% calculate assignment probabilities for each mixture component for all
% patches
GS2 = GS;
PYZ = zeros(GS.nmodels,size(Y,2));
for i=1:GS.nmodels
    GS2.covs(:,:,i) = GS.covs(:,:,i) + SigmaNoise;
    PYZ(i,:) = log(GS.mixweights(i)) + loggausspdf2(Y,GS2.covs(:,:,i));
end

% find the most likely component for each patch
[~,ks] = max(PYZ);

% and now perform weiner filtering
Xhat = zeros(size(Y));
for i=1:GS.nmodels
    inds = find(ks==i);
    Xhat(:,inds) = ((GS.covs(:,:,i)+SigmaNoise)\(GS.covs(:,:,i)*Y(:,inds) + SigmaNoise*repmat(GS.means(:,i),1,length(inds))));
end

% handle exclusion list stuff (inpainting only)
if ~isempty(excludeList)
    tt = T;
    tt(:,excludeList) = bsxfun(@plus,Xhat,meanY);
    Xhat = tt;
else
    Xhat = bsxfun(@plus,Xhat,meanY);
end
    
    
