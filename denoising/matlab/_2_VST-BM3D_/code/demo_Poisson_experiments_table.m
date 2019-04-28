% Script reproducing the results in Table 1 of the LNLA2009 paper [1].
%
% References:
% [1] M. Mäkitalo and A. Foi, "On the inversion of the Anscombe transformation in low-count Poisson image denoising", Proc. Int. Workshop on Local and Non-Local Approx. in Image Process., LNLA 2009, Tuusula, Finland, pp. 26-32, August 2009. doi:10.1109/LNLA.2009.5278406
% [2] M. Mäkitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
% [3] Zhang B., J.M. Fadili, and J-L. Starck, "Wavelets, ridgelets, and curvelets for Poisson noise removal", IEEE Trans. Image Process., vol. 17, no. 7, pp. 1093-1108, July 2008.
%
%  Alessandro Foi and Markku Mäkitalo - Tampere University of Technology - 2009
% -------------------------------------------------------------------------------

clear all

if exist('BM3D.m','file')
    warning('off','MATLAB:dispatcher:ShadowedMEXExtension')  %% disables warning about mex files shadowing dll files
else
   disp(' '),disp(' '),disp(' '),disp(' !!!  BM3D denoising software not found  !!!'),disp(' '),disp('     BM3D can be downloaded from http://www.cs.tut.fi/~foi/GCF-BM3D/ '),disp(' '),disp(' ')
   break 
end


load ./images/images.mat   % Load the five images used for the experiments reported in [3] and [1,2]. These images are kindly provided by the authors of [3].
                  % The image 'Galaxy' is copyright of Commissariat à l'Énergie Atomique (CEA) / Jean-Luc Starck, www.cea.fr, included here with permission.
                  % The image 'Cells' is originally from the ImageJ package http://rsb.info.nih.gov/ij (see http://rsb.info.nih.gov/ij/disclaimer.html).
y{1}=spots;
y{2}=galaxy;
y{3}=Ridges;
y{4}=Barbara;
y{5}=cells;
image_name{1}='Spots   ';
image_name{2}='Galaxy  ';
image_name{3}='Ridges  ';
image_name{4}='Barbara ';
image_name{5}='Cells   ';


disp('   ')
disp('----------------------------------------------------------------')
for jjj=1:5 %% loop on images
    randn('seed',0);  rand('seed',0);   %% fixes pseudo-random noise
    for jj=1:5   %% do five independent replications
        z=poissrnd(y{jjj});    %% generates Poisson-distributed observations (noisy data)
        [y_hat, PSNR_y_hat, NMISE_y_hat] = Poisson_denoising_Anscombe_exact_unbiased_inverse(z, y{jjj});   %%  denoise

        PSNR_y_hats(jj)=PSNR_y_hat;    %% stores PSNR and NMISE for each replication
        NMISE_y_hats(jj)=NMISE_y_hat;
    end
    meanNMISE_y_hats(jjj)=mean(NMISE_y_hats);    %% compute average NMISE
    disp(['Image: ',image_name{jjj},'   Average NMISE over ', num2str(jj),' realizations = ',num2str(meanNMISE_y_hats(jjj))]);
end
disp('----------------------------------------------------------------')
disp('   ')


%% end of code

