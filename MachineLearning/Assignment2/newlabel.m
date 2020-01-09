function [y_out, Clusterlabels]= newlabel(y_in,C,X,facit) %input: the previous classification vector we need to "desifer" and the clusters

Clusterlabels = zeros(1,size(C,2));
i = 1;
ii = 0;
iii = 0;                    % We initialize ii and iii to be a class that doesn't exist in order to assign 
while ii == 0 || iii == 0   % we need to find an example of a "1" and a "0" in our facit so we can identify our clusters
    if facit(i) == 0
        ii = i;
    elseif facit(i) == 1
        iii = i;
    end
    i = i+1;
end
X1 = X(:,ii);                   %example of a zero
X2 = X(:,iii);                  %example of a one
y_out = zeros(length(y_in),1);

%we want to compare each cluster to an example of a 1 and an example of a 0
%to see where the ressemblance is bigger
for i = 1:size(C,2)
    if fxdist(X1,C(:,i)) < fxdist(X2,C(:,i)) %meaning that: "if the cluster is more similar to a 0, set the label of cluster i as a 0"
        Clusterlabels(i) = 0;
    else
        Clusterlabels(i) = 1;
    end
end

for i = 1:length(y_out)
    y_out(i) = Clusterlabels(y_in(i));
end
end