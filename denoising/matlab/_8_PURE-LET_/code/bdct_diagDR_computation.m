%bdct_diagDR_computation: Monte-Carlo computation of diag{DR}, when D and R
%   are respectively the decomposition and the reconstruction matrix
%   associated with the overlapping BDCT.
%
%   See also BDCT_PURELET_denoise.
%
%   Author:
%   Florian Luisier
%   Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%
%   References:
%   [1] T. Blu, F. Luisier, "The SURE-LET Approach to Image Denoising,"
%   IEEE Transactions on Image Processing, vol. 16, no. 11, pp. 2778-2786,
%   November 2007.
%   [2] F. Luisier, "The SURE-LET Approach to Image Denoising," Swiss
%   Federal Institute of Technology Lausanne, EPFL Thesis no. 4566 (2010),
%   232 p., January 8, 2010.

clearvars;
restoredefaultpath;
path(path,'./transforms');
path(path,'./utilities');

% Choose a particular block size:
%--------------------------------
bsize = 12;
R     = dctmtx(bsize)/bsize;
D     = bsize*fliplr(R);

% Specify the size of your image (it should be divisible by 2 !):
%----------------------------------------------------------------
nx = 512;
ny = 512;

% Specify the number of realizations for the Monte-Carlo simulation:
%-------------------------------------------------------------------
NR = 1000;          % Great accuracy can be obtained with NR = 1000

% Specify the size of the mirror boundary extension (usually bsize):
%-------------------------------------------------------------------
E = bsize;

% Monte-Carlo computation of diag{DR}:
%-------------------------------------
nxe    = nx+2*E;
nye    = ny+2*E;
diagDR = single(zeros(nxe,nye*bsize^2));

h = waitbar(0,'Please wait...');
for k1 = 1:bsize
    for k2 = 1:bsize
        l = (k1-1)*bsize+k2;
        dr = zeros(nxe,nye);
        for i = 1:NR
            RandStream.setDefaultStream(RandStream('mt19937ar','seed',i-1));
            B  = randn(nxe,nye);
            Br = mex_recompose(B,R(k1,:),R(k2,:));
            Br = Br(E+1:end-E,E+1:end-E);
            Br = padarray(Br,[E,E],'symmetric','both');
            Bd = mex_decompose(Br,D(k1,:),D(k2,:));
            dr = dr+B.*Bd/NR;
            clear B*;
            waitbar(((l-1)*NR+i)/(bsize^2*NR),h);
        end
        diagDR(:,(l-1)*nye+1:l*nye) = single(dr);
    end
end
close(h);drawnow;

% Compress and Save "diagDR"
%---------------------------
diagDR = reshape(diagDR,bsize*nxe,bsize*nye);
[diagDR,rescale_params] = aux_compress(diagDR);
filename = ['./diagDR/BDCTdiagDR' num2str(nx) 'x' num2str(ny)...
            '_B' num2str(bsize) '_E' num2str(E)];
imwrite(diagDR,[filename '.jpg'],'jpg','quality',25);
save([filename '_params.mat'],'rescale_params');