function zBin=bin_B_h(z,h)
% basic binning function (v1.00)
%
% L. Azzari and A. Foi, "Variance Stabilization for Noisy+Estimate
% Combination in Iterative Poisson Denoising", submitted, March 2016
%
% http://www.cs.tut.fi/~foi/invansc/
%
%  L. Azzari and Alessandro Foi - Tampere University of Technology - 2016 - All rights reserved.
% -----------------------------------------------------------------------------------------------

if h>1

h=[h h];
hHalf  =  (h-double(mod(h,2)==1))/2;
modPad=h-mod(size(z)-1,h)-1;

% zBin = conv2(padarray(z,modPad,'symmetric','post'),ones(h),'same');
zBin = conv2(padarray(z,[modPad(1) 0 ],'symmetric','post'),ones(h(1),1),'same');
zBin = conv2(padarray(zBin,[0 modPad(2)],'symmetric','post'),ones(1,h(2)),'same');

% n_counter = conv2(padarray(ones(size(z)),modPad,0,'post'),ones(h),'same');  % how many pixels per bin? (may be different near boundaries)

% coordinates of bin centres
samples1  = hHalf(1)+double(mod(h(1),2)==1) : h(1) : size(zBin,1)-hHalf(1);
samples2  = hHalf(2)+double(mod(h(2),2)==1) : h(2) : size(zBin,2)-hHalf(2);

zBin=zBin(samples1,samples2);
else
    zBin=z;
end


return

