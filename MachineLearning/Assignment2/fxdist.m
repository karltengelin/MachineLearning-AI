function d = fxdist(x,C)
    k = size(C,2);
    d = zeros(1,k);
    for i = 1:k
        d(i) = sqrt(sum((C(:,i)-x).^2));
    end
end