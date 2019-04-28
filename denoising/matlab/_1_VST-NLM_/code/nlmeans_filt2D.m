function im_rec=nlmeans_filt2D(xn,sigmas,ksize,ssize,noise_std)

%%% 2D NL-means Filter.

%%% Author : B.K. SHREYAMSHA KUMAR
%%% Created on 06-05-2011.
%%% Updated on 06-05-2011. 


half_ksize=floor(ksize/2);
half_ssize=floor(ssize/2);

%%% To take care of boundaries.
[M,N]=size(xn);
xm=zeros(M+ssize-1,N+ssize-1);
xm(half_ssize+1:M+half_ssize,half_ssize+1:N+half_ssize)=xn;
xm(1:half_ssize,:)=xm(ssize:-1:half_ssize+2,:);
xm(M+half_ssize+1:M+ssize-1,:)=xm(M+half_ssize-1:-1:M,:);
xm(:,1:half_ssize)=xm(:,ssize:-1:half_ssize+2);
xm(:,N+half_ssize+1:N+ssize-1)=xm(:,N+half_ssize-1:-1:N);

%%% Gaussian Kernel Generation. 
gauss_win=gauss_ker2D(sigmas,ksize);
% gauss_win=ones(ksize,ksize);

%%% NL-means Filter Implementation.
filt_h=0.55*noise_std;
[M,N]=size(xm);
for ii=half_ssize+1:M-half_ssize
   for jj=half_ssize+1:N-half_ssize
      xtemp=xm(ii-half_ksize:ii+half_ksize,jj-half_ksize:jj+half_ksize);      
      search_win=xm(ii-half_ssize:ii+half_ssize,jj-half_ssize:jj+half_ssize);
      for kr=1:(ssize-ksize+1)
         for kc=1:(ssize-ksize+1)   
% % %             dist=gauss_win.*(xtemp-search_win(kr:kr+ksize-1,kc:kc+ksize-1));
% % %             sq_dist=sum(sum((dist.^2)))/(ksize^2);
            euclid_dist=(xtemp-search_win(kr:kr+ksize-1,kc:kc+ksize-1)).^2;
            wt_dist=gauss_win.*euclid_dist;
            sq_dist=sum(sum((wt_dist)))/(ksize^2);
            weight(kr,kc)=exp(-max(sq_dist-(2*noise_std^2),0)/filt_h^2);
         end
      end
      sum_wt=sum(sum(weight));
      weightn=weight/sum_wt;
      sum_pix=sum(sum(search_win(half_ksize+1:ssize-half_ksize,half_ksize+1:ssize-half_ksize).*weightn));
      im_rec(ii-half_ssize,jj-half_ssize)=sum_pix;
   end
end