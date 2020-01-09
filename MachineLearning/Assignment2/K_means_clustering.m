    function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;
%%
for kiter = 1:intermax

    % Step 1: Assign to clusters
    for i = 1:N
        d = fxdist(X(:,i),C);
        y(i) = find(d == min(d));
    end
    % Step 2: Assign new clusters
    for i = 1:size(C,2)
        Nk = sum(y==i);
        C(:,i) = Nk\sum(X(:,y==i),2);
    end
        
    if fcdist(C,Cold) < conv_tol
        return
    end
    Cold = C;
end

end

function d = fxdist(x,C)
    k = size(C,2);
    d = zeros(1,k);
    for i = 1:k
        d(i) = sqrt(sum((C(:,i)-x).^2));
    end
end

function d = fcdist(C1,C2)
    d = sqrt(sum((C1-C2).^2));
end
