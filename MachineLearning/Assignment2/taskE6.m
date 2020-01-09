load('A2_data.mat')

X_train = train_data_01';
T_train = train_labels_01;

model = fitcsvm(X_train,T_train);

% X_test = test_data_01';
% T_test = test_labels_01;
X_test = train_data_01';
T_test = train_labels_01;

[svm_label,~]= predict(model,X_test);

find(svm_label ~= T_test)
sum(svm_label ~= T_test)

predicted_ones = find(svm_label == 1);
predicted_zeros = find(svm_label == 0);

nbr_of_ones_in_ones = 0;
nbr_of_zeros_in_ones = 0;
nbr_of_ones_in_zeros = 0;
nbr_of_zeros_in_zeros = 0;

for i = 1:length(predicted_ones)
    if 1 == T_test(predicted_ones(i))
        nbr_of_ones_in_ones = nbr_of_ones_in_ones +1;
    else
        nbr_of_zeros_in_ones = nbr_of_zeros_in_ones +1;
    end   
end
for i = 1:length(predicted_zeros)
    if 0 == T_test(predicted_zeros(i))
        nbr_of_zeros_in_zeros = nbr_of_zeros_in_zeros +1;
    else
        nbr_of_ones_in_zeros = nbr_of_ones_in_zeros +1;
    end   
end

