function d = fcdist(C1,C2)
    d = sqrt(sum((C1-C2).^2));
end