function y = loggausspdf2(X, sigma)
% log pdf of Gaussian with zero mena
% Based on code written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
d = size(X,1);

[R,p]= chol(sigma);
if p ~= 0
    error('ERROR: sigma is not SPD.');
end
q = sum((R'\X).^2,1);  % quadratic term (M distance)
c = d*log(2*pi)+2*sum(log(diag(R)));   % normalization constant
y = -(c+q)/2;
