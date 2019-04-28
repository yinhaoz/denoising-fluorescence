function [cleanZ] = PatchDCTGG(Z,patchSize,noiseSD,imsize,W,invW,excludeList)
% a simple thresholding denoiser - approx. corresponds to a sprase prior
% over the marginals
if ~exist('excludeList','var')
    excludeList = [];
end

meanZ = mean(Z);
Z = bsxfun(@minus,Z,meanZ);
if (~isempty(excludeList))
    WZ = W*Z(:,excludeList);
else
    WZ = W*Z;
end

t = noiseSD*3;
WZ(abs(WZ)<t)=0;
cleanZ = Z;
if ~isempty(excludeList)
    cleanZ(:,excludeList) = invW*WZ;
else
    cleanZ = invW*WZ;
end
cleanZ = bsxfun(@plus,cleanZ,meanZ);
