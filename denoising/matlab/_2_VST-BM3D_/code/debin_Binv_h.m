function y_j=debin_Binv_h(yBin,size_z,h,niter)
% Debinning function (v1.00)
%
% L. Azzari and A. Foi, "Variance Stabilization for Noisy+Estimate
% Combination in Iterative Poisson Denoising", submitted, March 2016
%
% http://www.cs.tut.fi/~foi/invansc/
%
%  L. Azzari and Alessandro Foi - Tampere University of Technology - 2016 - All rights reserved.
% -----------------------------------------------------------------------------------------------


if h>1
    
    % binning
    h=[h h];
    hHalf  =  (h-double(mod(h,2)==1))/2;
    modPad=h-mod(size_z-1,h)-1;
    
    % how many pixels per bin? (may be different near boundaries)
    % n_counter = conv2(padarray(ones(size_z),modPad,'symmetric','post'),ones(h),'same');
    n_counter = conv2(padarray(ones(size_z),[modPad(1) 0 ],'symmetric','post'),ones(h(1),1),'same');
    n_counter = conv2(padarray(n_counter,[0 modPad(2)],'symmetric','post'),ones(1,h(2)),'same');
    
    % coordinates of bin counts
    x1c  = hHalf(1)+double(mod(h(1),2)==1)+[0 : size(yBin,1)-1]*h(1);
    x2c  = hHalf(2)+double(mod(h(2),2)==1)+[0 : size(yBin,2)-1]*h(2);
    
    % coordinates of bin centers
    x1  = hHalf(1)+1-double(mod(h(1),2)==0)/2+[0 : size(yBin,1)-1]*h(1);
    x2  = hHalf(2)+1-double(mod(h(2),2)==0)/2+[0 : size(yBin,2)-1]*h(2);
    
    % coordinates of image pixels
    ix1 = 1 : size_z(1);
    ix2 = 1 : size_z(2);
    
    y_j=0;
    for jj=1:max(1,niter);
        
        % residual
        if jj>1
            r_j=yBin-bin_B_h(y_j,h(1));
            % disp(num2str(max(abs(r_j(:)))));  print out maximum of residual, to show convergence
        else
            r_j=yBin;
        end
        
        % interpolation
        y_j=max(0,y_j+interp2(x2',x1,r_j./n_counter(x1c,x2c),ix2',ix1,'spline'));
        
    end
else
    
    y_j=yBin;
    
end

return