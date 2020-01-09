function lab = correctlabel(X,C,labels) %we input the training data, the produced clusters and the correct labeling
lab = zeros(length(X),1);

for i = 1:length(X)
    lab(i) = K_means_classifier(X(:,i),C);
end

right = 0;
wrong = 0;

for i = 1:length(X)
    if lab(i) == labels(i)
        right = right + 1;
    else
        wrong = wrong + 1;
    end
end

if right < wrong
    lab = abs(lab - ones(length(X),1));
end

% we check how many missclassifications occur for train data
%misclass_train = find(lab ~= labels);
%nbr_of_misclass = length(misclass_train);



end