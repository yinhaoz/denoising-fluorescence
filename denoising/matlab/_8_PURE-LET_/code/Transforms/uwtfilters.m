function [D,R] = uwtfilters(wtype,it)
%   Author: 
%   Florian Luisier
%   School of Engineering and Applied Sciences
%   Harvard University
%   Cambridge, MA 02138, USA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Ld,Hd,Lr,Hr] = wfilters(wtype);
D = cell(it,1);
R = cell(it,1);
D{1}{1} = Ld; 
D{1}{2} = Hd; 
R{1}{1} = Lr/2; 
R{1}{2} = Hr/2; 
for i = 2:it
    K  = 2^(i-1);
    Ld2 = upsample(Ld,K);
    Hd2 = upsample(Hd,K);
    D{i}{1} = conv(Ld2,D{i-1}{1});
    D{i}{2} = conv(Hd2,D{i-1}{1});
    D{i}{1} = D{i}{1}(1:end-K+1); 
    D{i}{2} = D{i}{2}(1:end-K+1); 
    Lr2 = upsample(Lr/2,K);
    Hr2 = upsample(Hr/2,K);
    R{i}{1} = conv(Lr2,R{i-1}{1});
    R{i}{2} = conv(Hr2,R{i-1}{1});
    R{i}{1} = R{i}{1}(1:end-K+1); 
    R{i}{2} = R{i}{2}(1:end-K+1);
end