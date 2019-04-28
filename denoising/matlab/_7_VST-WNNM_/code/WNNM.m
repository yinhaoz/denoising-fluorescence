
function  [X] =  WNNM( Y, C, NSig, m, Iter )
    [U,SigmaY,V] =   svdecon(full(Y));    
    PatNum       = size(Y,2);
    TempC  = C*sqrt(PatNum)*2*NSig^2;
    [SigmaX,svp] = ClosedWNNM(SigmaY,TempC,eps);                        
    X =  U(:,1:svp)*diag(SigmaX)*V(:,1:svp)' + m;     
return;
