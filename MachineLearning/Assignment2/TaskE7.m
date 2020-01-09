load('A2_data.mat')

X_train = train_data_01';
T_train = train_labels_01;
beta_max = 10;
total_missclass = zeros(1,2*beta_max-1);
index = 1;

for beta = 1 %1:0.5:beta_max    
model = fitcsvm(X_train,T_train,'KernelFunction','gaussian', 'KernelScale',beta);

% X_test = test_data_01';
% T_test = test_labels_01;
X_test = train_data_01';
T_test = train_labels_01;

[svm_label,~]= predict(model,X_test);

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

total_missclass(index) = nbr_of_ones_in_zeros + nbr_of_zeros_in_ones;
index = index + 1;
end

x_axis = (1:0.5:beta_max);

figure
plot(x_axis,total_missclass)
xlabel('Beta')
ylabel('Missclassifications')

Answer_to_task = [0 1; nbr_of_zeros_in_zeros nbr_of_ones_in_zeros; nbr_of_zeros_in_ones nbr_of_ones_in_ones];
