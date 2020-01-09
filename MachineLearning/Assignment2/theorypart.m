K = [20 6 2 12; 6 2 0 2; 2 0 2 6; 12 2 6 20];
y = [1 -1 -1 1];

for i = 1:length(y)
    for ii = 1:length(y)
        X(i,ii) = y(i)*y(ii)*K(i,ii);
    end
end

summary = sum(sum(X))

