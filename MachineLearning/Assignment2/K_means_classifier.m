function lab = K_means_classifier(x,C)
    k = size(C,2);
    dist = zeros(1,k);
for i = 1:k
    dist(i) = sqrt(sum((C(:,i)-x).^2));
end
lab = find(dist == min(dist));