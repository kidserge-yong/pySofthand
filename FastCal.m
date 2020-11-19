
function [U,V,CPUtime,NRV,RRV] = FastCal(X,V,iter)

%%Input 
%% X = M (number of sample) x N (number of channel) matrix
%% U = M (number of sample) x K (number of component) matrix
%% V = N (number of channel) x K (number of component) matrix
%% iter: #iterations

%%Output
%% CPUtime: CPUtime taken for each update
%% NRV    : Normalized Residual Value for each update
%%             || X-UV^T || / ||X||
%% RRV    : Relative Residual Value for each update
%%            log_{10} (||X-U_{t}V_{t}|| / ||X-U_{0}V_{0}||)


%% initialization
K = size(V,2);
M = size(X,1);
U = rand(M, K);
CPUtime = zeros(iter+1,1);
NRV = zeros(iter+1,1);
RRV = zeros(iter+1,1);
eps = 1e-08;

normX = norm(X,'fro')^2;

Ap0 = normX - 2*trace( (U'*X)*V ) + trace( (U'*U)*(V'*V) );

CPUtime(1)=0;
NRV(1) = Ap0 ./ normX;
RRV(1) = 1;

for loop=1:iter
    
    t=cputime; 
    
    A = X*V;
    B = V'*V;
    
    for k = 1:K
        tmp = (A(:,k)-(U * B(:,k)) + (U(:,k)*B(k,k)))./B(k,k);
        tmp(tmp<=eps)=eps;
        U(:,k)=tmp;
    end
    
    CPUtime(loop)=cputime-t;
    %% measuring
    Ap=normX-2*trace( (U'*X)*V ) + trace( (U'*U)*(V'*V) );
    NRV(loop+1) = Ap ./normX;
    RRV(loop+1) = log10(Ap./ Ap0);
end