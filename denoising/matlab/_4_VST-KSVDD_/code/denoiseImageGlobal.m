function [IOut,output] = denoiseImageGlobal(Image, sigma, varargin)
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   G L O B A L 
%           ( O R   G I V E N )   D I C T I O N A R Y
%==========================================================================
% function IOut = denoiseImageGlobal(Image, sigma, varargin)
% denoise an image by training a dictionary on patches from the noisy image,  
% sparsely representing each block with this dictionary and averaging the represented parts.
% Detailed description can be found in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
% The trained dictionary is given in a specific file
% 'globalTrainedDictionary.mat' that must accompany this function.
% The user may also specify a different dictionary to be used by using the
% optional arguments.
% ===================================================================
% INPUT ARGUMENTS : Image - the noisy image (gray-level scale)
%                   sigma - the s.d. of the noise (assume to be white Gaussian).
%    Optional arguments:              
%                  'blockSize' - the size of the blocks the algorithm
%                       works. All blocks are squares, therefore the given
%                       parameter should be one number (width or height).
%                       Default value: 8.
%                  'errorFactor' - a factor that multiplies sigma in order
%                       to set the allowed representation error. In the
%                       experiments presented in the paper, it was set to 1.15
%                       (which is also the default  value here).
%                  'maxBlocksToConsider' - maximal number of blocks that
%                       can be processed. This number is dependent on the memory
%                       capabilities of the machine, and performances'
%                       considerations. If the number of available blocks in the
%                       image is larger than 'maxBlocksToConsider', the sliding
%                       distance between the blocks increases. The default value
%                       is: 250000.
%                  'slidingFactor' - the sliding distance between processed
%                       blocks. Default value is 1. However, if the image is
%                       large, this number increases automatically (because of
%                       memory requirements). Larger values result faster
%                       performances  (because of fewer processed blocks).
%                  'givenDictionary' - a different dictionary to consider
%                       for denoising. The user is responsible that the number
%                       of rows in the dictionary will resemble the dimension of
%                       the blocks.
%                  'waitBarOn' - can be set to either 1 or 0. If
%                       waitBarOn==1 a waitbar, presenting the progress of the
%                       algorithm will be displayed.
% OUTPUT ARGUMENTS : Iout - a 2-dimensional array in the same size of the
%                   input image, that contains the cleaned image.
%                    output - a struct that contains that following field:
%                       D - the dictionary used for denoising
% =========================================================================

fildNameForGlobalDictionary = 'globalTrainedDictionary';
Reduce_DC = 1;
[NN1,NN2] = size(Image);
waitBarOn = 1;
C = 1.15;
maxBlocksToConsider = 260000;
slidingDis = 1;
bb = 8;
givenDictionaryFlag = 0;
for argI = 1:2:length(varargin)
    if (strcmp(varargin{argI}, 'slidingFactor'))
        slidingDis = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'errorFactor'))
        C = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'maxBlocksToConsider'))
        maxBlocksToConsider = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'givenDictionary'))
        D = varargin{argI+1};
        givenDictionaryFlag = 1;
    end
    if (strcmp(varargin{argI}, 'blockSize'))
        bb = varargin{argI+1};
    end
    if (strcmp(varargin{argI}, 'waitBarOn'))
        waitBarOn = varargin{argI+1};
    end
end
if (~givenDictionaryFlag)
    eval(['load ',fildNameForGlobalDictionary]);
    D = currDictionary;
end
errT = C*sigma;
%blocks = im2col(Image,[NN1,NN2],[bb,bb],'sliding');
while (prod(size(Image)-bb+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(Image,[bb,bb],slidingDis);

if (waitBarOn)
    counterForWaitBar = size(blocks,2);
    h = waitbar(0,'Denoising In Process ...');
end

% go with jumps of 10000
for jj = 1:10000:size(blocks,2)
    if (waitBarOn)
        waitbar(jj/counterForWaitBar);
    end
    jumpSize = min(jj+10000-1,size(blocks,2));
    if (Reduce_DC)
        vecOfMeans = mean(blocks(:,jj:jumpSize));
        blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
    end
    %Coefs = mexOMPerrIterative(blocks(:,jj:jumpSize),D,errT);
    Coefs = OMPerr(D,blocks(:,jj:jumpSize),errT);
    if (Reduce_DC)
        %block=reshape(D*a+mm+MM,[bb,bb]);
        blocks(:,jj:jumpSize)= D*Coefs + ones(size(blocks,1),1) * vecOfMeans;
    else
        blocks(:,jj:jumpSize)= D*Coefs ;
    end
end

count = 1;
Weight=zeros(NN1,NN2);
IMout = zeros(NN1,NN2);
[rows,cols] = ind2sub(size(Image)-bb+1,idx);
for i  = 1:length(cols)
    col = cols(i); row = rows(i);
    block =reshape(blocks(:,count),[bb,bb]);
    IMout(row:row+bb-1,col:col+bb-1)=IMout(row:row+bb-1,col:col+bb-1)+block;
    Weight(row:row+bb-1,col:col+bb-1)=Weight(row:row+bb-1,col:col+bb-1)+ones(bb);
    count = count+1;
end;

if (waitBarOn)
    close(h);
end
output.D = D;
IOut = (Image+0.034*sigma*IMout)./(1+0.034*sigma*Weight);

