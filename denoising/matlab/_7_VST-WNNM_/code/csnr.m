function s=csnr(A,B,row,col)

[n,m,ch]=size(A);

if ch==1
   e=A-B;
   e=e(row+1:n-row,col+1:m-col);
   me=mean(mean(e.^2));
   s=10*log10(255^2/me);
else
   e=A-B;
   e=e(row+1:n-row,col+1:m-col,:);
   e1=e(:,:,1);e2=e(:,:,2);e3=e(:,:,3);
   me1=mean(mean(e1.^2));
   me2=mean(mean(e2.^2));
   me3=mean(mean(e3.^2));
   mse=(me1+me2+me3)/3;
   s  = 10*log10(255^2/mse);
%    s(1)=10*log10(255^2/me1);
%    s(2)=10*log10(255^2/me2);
%    s(3)=10*log10(255^2/me3);
end


return;