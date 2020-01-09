load('A2_data.mat')
train = normalize(train_data_01,'center');
test = normalize(test_data_01,'center');
img = reshape(train(:,1),[28 28]);
imagesc(img)
%% Here we plot what the clusters really look like by using PCA, which utilizes SVD
d = 2;
[U,S,V] = svd(train);
U_short = U(:,1:d);
Z = U_short'*train;

figure
hold on
for i = 1:length(train_labels_01)
    if train_labels_01(i) == 1
        plot(Z(1,i),Z(2,i),'r+')
    else
        plot(Z(1,i),Z(2,i),'b*')
    end
end
legend('set 1 - ones', 'set 2 - zeros')
hold off

%% From here we use our machine learning methods (clustering)
% first K = 2
K = 2;
[y_train,C1] = K_means_clustering(train,K);
y_train = correctlabel(train,C1,train_labels_01); % this we do in order to get 0 and 1 instead of 1 and 2 in y_train
% however we do not know if a "1" in y_train represents a "1" in the training data and vice versa
%therefore we do a test to see if the label is labeling correctly


[U2,S2,V2] = svd(train);
U2_short = U2(:,1:d);
Z2 = U2_short'*train;

figure
hold on
for i = 1:length(y_train)
    if y_train(i) == 1
        plot(Z2(1,i),Z2(2,i),'r+')
    else
        plot(Z2(1,i),Z2(2,i),'b*')
    end
end
legend('set 1 - ones', 'set 2 - zeros')
hold off
%% Here we want to see what the clusters C look like
img1 = reshape(C1(:,1),[28 28]);
img2 = reshape(C1(:,2),[28 28]);
figure
subplot(121)
imagesc(img1)
subplot(122)
imagesc(img2)

%% Now K = 5
K = 5;
[y_train,C_train] = K_means_clustering(train,K);
y_train_old = y_train;
% now y_train contains 5 classes, meaning that we need to interpret the
% output in order to match 2 classes (we only have 2 classes, 0 and 1)

%we know from train_labels that the first column vector is classified as a
%0 and the second as a 1. We use this information in order to classify the
%clusters

Clusterlabels = zeros(1,K);
X1 = train(:,1);
X2 = train(:,2);

for i = 1:size(C_train,2)
    if fxdist(X1,C_train(:,i)) < fxdist(X2,C_train(:,i))%meaning that: "if the cluster is more similar to a 0, set the label of cluster i as a 0"
        Clusterlabels(i) = 0;
    else
        Clusterlabels(i) = 1;
    end
end

for i = 1:length(y_train)
    y_train(i) = Clusterlabels(y_train(i));
end

[U2,S2,V2] = svd(train);
U2_short = U2(:,1:d);
Z2 = U2_short'*train;

figure
hold on
for i = 1:length(y_train_old)
    if y_train_old(i) == 5
        set5 = plot(Z2(1,i),Z2(2,i),'r+');
        set5;
    elseif y_train_old(i) == 4
        set4 = plot(Z2(1,i),Z2(2,i),'b*');
        set4;
    elseif y_train_old(i) == 3
        set3 = plot(Z2(1,i),Z2(2,i),'cd');
        set3;
    elseif y_train_old(i) == 2
        set2 = plot(Z2(1,i),Z2(2,i),'g^');
        set2;
    elseif y_train_old(i) == 1
        set1 = plot(Z2(1,i),Z2(2,i),'mx');
        set1;
    end
end
legend([set5 set4 set3 set2 set1],{'set 5','set 4','set 3','set 2','set 1'})
hold off
%% How does the clusters look:
img1 = reshape(C_train(:,1),[28 28]);
img2 = reshape(C_train(:,2),[28 28]);
img3 = reshape(C_train(:,3),[28 28]);
img4 = reshape(C_train(:,4),[28 28]);
img5 = reshape(C_train(:,5),[28 28]);
figure
subplot(151)
imagesc(img1)
subplot(152)
imagesc(img2)
subplot(153)
imagesc(img3)
subplot(154)
imagesc(img4)
subplot(155)
imagesc(img5)

%% E4 Now we want to classify using clustering
% first we need to look in the previous pictures to see which cluster is a
% one and which is a zero
X = train;

[y_train,C_train] = K_means_clustering(X,K);
distance = zeros(1,size(C_train,2));
C_label = zeros(1,K);

for i = 1:length(X)
    for ii = 1:size(C_train,2)
        distance(ii) = fxdist(X(:,i),C_train(:,ii));
    end
    mindist = find(distance == min(distance));
    C_label(mindist) = train_labels_01(i);
end

%%

%%%%%%%%%% In this section we make sure that the label is correct, i.e a 1
%%%%%%%%%% is labeled as a 1 and a 0 as a 0, here for train
lab_train = zeros(length(X),1);

for i = 1:length(X)
    lab_train(i) = K_means_classifier(X(:,i),C_train);
end

right = 0;
wrong = 0;

for i = 1:length(X)
    if lab_train(i) == train_labels_01(i)
        right = right + 1;
    else
        wrong = wrong + 1;
    end
end

if right < wrong
    lab_train = abs(lab_train - ones(length(X),1));
end

%we check how many missclassifications occur for train data
misclass_train = find(lab_train ~= train_labels_01);
nbr_of_misclass_train = length(misclass_train);

%%%%%%%%%% In this section we make sure that the label is correct, i.e a 1
%%%%%%%%%% is labeled as a 1 and a 0 as a 0, now for test
Y = test;

[y_test,C_test] = K_means_clustering(Y,K);
lab_test = zeros(length(Y),1);

for i = 1:length(Y)
    lab_test(i) = K_means_classifier(Y(:,i),C_test);
end

right1 = 0;
wrong1 = 0;

for i = 1:length(Y)
    if lab_test(i) == test_labels_01(i)
        right1 = right1 + 1;
    else
        wrong1 = wrong1 + 1;
    end
end

if right1 < wrong1
    lab_test = abs(lab_test - ones(length(Y),1));
end

%we check how many missclassifications occur for test data
misclass_test = find(lab_test ~= test_labels_01);
nbr_of_misclass1 = length(misclass_test);
%Now we have made sure that our labeling is labeling correct
%%













