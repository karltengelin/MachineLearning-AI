load('A2_data.mat')
Norm_train = mean(train_data_01,2);
Norm_test = mean(test_data_01,2);
train = train_data_01-Norm_train;
test = test_data_01-Norm_test;
%% assigning which data we are interested in (train or test as X) and how many clusters we assume there are (K)
X = train;
facit = train_labels_01;
K = 5;

nbr_of_zeros = zeros(K,1);          %initiating the vector in which we will store number of zeros
nbr_of_ones = zeros(K,1);           %initiating the vector in which we will store number of ones
d = 2;                              %what dimension we are projecting onto
% keys = cell(1,K);                   %initiating an array in which we will store our keys to desciffer the centroid classes
% Kalle = cell(1,K);
%% Here we plot what the clusters really look like by using PCA, which utilizes SVD, and the correct labeling (facit)
[U,~,~] = svd(X);
U_short = U(:,1:d);
Z = U_short'*X;

clear U U_short

figure
hold on
for i = 1:length(facit)
    if facit(i) == 1
        plot(Z(1,i),Z(2,i),'r+')
    else
        plot(Z(1,i),Z(2,i),'b*')
    end
end
legend('Class 0', 'Class 1')
hold off

%% From here on we use our machine learning methods (clustering)
C_nbr_1 = zeros(1,K);                               %these vectors will be used to store number of ones and zeros in each cluster
C_nbr_0 = zeros(1,K);
    
    [y,C] = K_means_clustering(X,K);                %creating a "class vector" y and the centriods C
    [y_new, clusterkey] = newlabelv2(y,C,X,facit);  %creating a new class vector y_new where whe have interpreted the values in y (y contains 1,2,3...,K and we want to know if class "1" in y is actually class 0 or 1 and so on)
    C = C + Norm_train;                             % we re-add the mean which we took away for the pca in order to iluustrate clearer centroid images
    
    keys = [1:K ; clusterkey];                      %this shows how one should interpret the clusterclasses as zeros and ones. so for example if the key is [1 2 3;0 1 1] that means that clusterclass 1 in y is actually class 0, 2 is 1 and 3 is 1
                                                    % i.e it shows which cluster is assigned to which class                                                 
    for zz = 1:length(C_nbr_1)                      % here we want to check how many elements in each cluster which is labeled as a ''1'' 
                                                    % and how many is labeled as a ''0''
        for i = 1:length(y)                         %Observe that this is essentially the K_means_classifier that we were supposed to implement, however used restrospectively
            if y(i) == zz && facit(i) == 1
            C_nbr_1(zz) = C_nbr_1(zz) + 1;
            elseif y(i) == zz
            C_nbr_0(zz) = C_nbr_0(zz) + 1;
            end
        end
    end
                                                    
    [U,~,~] = svd(X);                               %performing SVD

    U_short = U(:,1:d);
    Z = U_short'*X;                                 %2d projection

    clear U U_short

    figure                                          %plotting the clustering based on the classes derived in y
    hold on
    for i = 1:length(y)
        if y(i) == 5                                %we plot here for a maximum of 5 clusters, therefore we assume there is no element in y larger than 5
            set5 = plot(Z(1,i),Z(2,i),'g^');
            set5;
        elseif y(i) == 4
            set4 = plot(Z(1,i),Z(2,i),'mx');
            set4;
        elseif y(i) == 3
            set3 = plot(Z(1,i),Z(2,i),'cd');
            set3;
        elseif y(i) == 2
            set2 = plot(Z(1,i),Z(2,i),'b*');
            set2;
        elseif y(i) == 1
            set1 = plot(Z(1,i),Z(2,i),'r+');
            set1;
        end
    end
    legend([set5 set4 set3 set2 set1],{'Cluster 5','Cluster 4','Cluster 3','Cluster 2','Cluster 1'})
    %legend([set2 set1],{'Cluster 2','Cluster 1'})
    clear set5 set4 set3 set2 set1
    hold off

    % Here we want to see what the clusters C look like:
%     if K < 2
%         img1 = reshape(C(:,1),[28 28]);
%         elseif K < 3
%             img1 = reshape(C(:,1),[28 28]);
%             img2 = reshape(C(:,2),[28 28]);
%         elseif K < 4     
%             img1 = reshape(C(:,1),[28 28]);
%             img2 = reshape(C(:,2),[28 28]);
%             img3 = reshape(C(:,3),[28 28]);
%         elseif K < 5
%             img1 = reshape(C(:,1),[28 28]);
%             img2 = reshape(C(:,2),[28 28]);
%             img3 = reshape(C(:,3),[28 28]);
%             img4 = reshape(C(:,4),[28 28]);
%         elseif K < 6
%             img1 = reshape(C(:,1),[28 28]);
%             img2 = reshape(C(:,2),[28 28]);
%             img3 = reshape(C(:,3),[28 28]);
%             img4 = reshape(C(:,4),[28 28]);
%             img5 = reshape(C(:,5),[28 28]);
%     end
Missclassification_per_cluster = [1:K; C_nbr_1; C_nbr_0]
clusterkey


% figure
% subplot(151)
% imagesc(img1)
% title('Cluster 1')
% subplot(152)
% imagesc(img2)
% title('Cluster 2')
% subplot(153)
% imagesc(img3)
% title('Cluster 3')
% subplot(154)
% imagesc(img4)
% title('Cluster 4')
% subplot(155)
% imagesc(img5)
% title('Cluster 5')












