%uwt_diagDR_computation: Monte-Carlo computation of diag{DR}, when D and R
%   are respectively the decomposition and the reconstruction matrix
%   associated with the undecimated wavelet transform.
%
%   See also UWT_PURELET_denoise, DICT_PURELET_denoise.
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

% Choose a particular orthonormal wavelet filter:
%------------------------------------------------
wtype = 'haar';     % Haar is usually the best performer

% Specify the size of your image (it should be divisible by 2 !):
%----------------------------------------------------------------
nx = 256;
ny = 256;

% Specify the number of realizations for the Monte-Carlo simulation:
%-------------------------------------------------------------------
NR = 1000;          % Great accuracy can be obtained with NR = 1000

% Specify the number of decomposition level for the UWT:
%-------------------------------------------------------
J  = aux_num_of_iters([nx,ny]);
IT = min(J);
[D,R] = uwtfilters(wtype,IT);

% Specify the size of the mirror boundary extension (usually 2^IT):
%------------------------------------------------------------------
E = 2^IT;

% Monte-Carlo computation of diag{DR}:
%-------------------------------------
nxe    = nx+2*E;
nye    = ny+2*E;
diagDR = zeros(nxe,3*IT*nye);

h = waitbar(0,'Please wait...');
L = 0;
for i = 1:IT
    for o1 = 1:2
        for o2 = 1:2
            if(o1==1 && o2==1)
                % Lowpass at scale i
                continue;
            end
            L  = L+1;
            D1 = D{i}{o1};
            D2 = D{i}{o2};
            R1 = R{i}{o1};
            R2 = R{i}{o2};
            dr = 0;
            for r = 1:NR
                s = RandStream('mt19937ar','seed',r-1);
                RandStream.setDefaultStream(s);
                B  = randn(nxe,nye);
                B  = B/std(B(:));
                Br = mex_recompose(B,R1,R2);
                Br = Br(E+1:end-E,E+1:end-E);
                Br = padarray(Br,[E,E],'symmetric','both');
                Bd = mex_decompose(Br,D1,D2);
                dr = dr+B.*Bd/NR;
                clear B*;
                waitbar(((L-1)*NR+r)/(3*IT*NR),h);
            end
            diagDR(:,(L-1)*nye+1:L*nye) = dr;
        end
    end
end
close(h);

% Compress and Save "diagDR"
%---------------------------
[diagDR,rescale_params] = aux_compress(diagDR);
filename = ['./diagDR/UWTdiagDR' num2str(nx) 'x' num2str(ny) '_' wtype...
            '_E' num2str(E)];
imwrite(diagDR,[filename '.jpg'],'jpg','quality',90);
save([filename '_params.mat'],'rescale_params');