function row = zerow(A)
    k = 1;
    for i = 1:size(A)
        if A(i,1) == 0 && A(i,2) == 0 && A(i,3) == 0
            row(k) = i;
            k = k+1;
        else
            
        end
    end
end