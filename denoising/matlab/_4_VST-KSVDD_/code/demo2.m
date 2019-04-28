%============================================================
%               demo2 - denoise an image
% this is a run_file the demonstrate how to denoise an image, 
% using dictionaries. The methods implemented here are the same
% one as described in "Image Denoising Via Sparse and Redundant
% representations over Learned Dictionaries", (appeared in the 
% IEEE Trans. on Image Processing, Vol. 15, no. 12, December 2006).
%============================================================

clear
bb=8; % block size
RR=4; % redundancy factor
K=RR*bb^2; % number of atoms in the dictionary

sigma = 25; 
pathForImages ='';
imageName = 'barbara.png';
[IMin0,pp]=imread(strcat([pathForImages,imageName]));
IMin0=im2double(IMin0);
if (length(size(IMin0))>2)
    IMin0 = rgb2gray(IMin0);
end
if (max(IMin0(:))<2)
    IMin0 = IMin0*255;
end

IMin=IMin0+sigma*randn(size(IMin0));
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)));
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   O V E R C O M P L E T E 
%                        D C T    D I C T I O N A R Y
%==========================================================================
[IoutDCT,output] = denoiseImageDCT(IMin, sigma, K);

PSNROut = 20*log10(255/sqrt(mean((IoutDCT(:)-IMin0(:)).^2)));
figure;
subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
subplot(1,3,2); imshow(IMin,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
subplot(1,3,3); imshow(IoutDCT,[]); title(strcat(['Clean Image by DCT dictionary, ',num2str(PSNROut),'dB']));
figure;
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb,0);
title('The DCT dictionary');
%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   G L O B A L 
%           ( O R   G I V E N )   D I C T I O N A R Y
%==========================================================================
[IoutGlobal,output] = denoiseImageGlobal(IMin, sigma,K);

PSNROut = 20*log10(255/sqrt(mean((IoutGlobal(:)-IMin0(:)).^2)));
figure;
subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
subplot(1,3,2); imshow(IMin,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
subplot(1,3,3); imshow(IoutGlobal,[]); title(strcat(['Clean Image by Global Trained dictionary, ',num2str(PSNROut),'dB']));
figure;
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);
title('The dictionary trained on patches from natural images');

%==========================================================================
%   P E R F O R M   D E N O I S I N G   U S I N G   A   D I C T  I O N A R Y
%                  T R A I N E D   O N   N O I S Y   I M A G E
%==========================================================================
[IoutAdaptive,output] = denoiseImageKSVD(IMin, sigma,K);

PSNROut = 20*log10(255/sqrt(mean((IoutAdaptive(:)-IMin0(:)).^2)));
figure;
subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
subplot(1,3,2); imshow(IMin,[]); title(strcat(['Noisy image, ',num2str(PSNRIn),'dB']));
subplot(1,3,3); imshow(IoutAdaptive,[]); title(strcat(['Clean Image by Adaptive dictionary, ',num2str(PSNROut),'dB']));

figure;
I = displayDictionaryElementsAsImage(output.D, floor(sqrt(K)), floor(size(output.D,2)/floor(sqrt(K))),bb,bb);
title('The dictionary trained on patches from the noisy image');