function matrix=npermutek2(n,k,stop)

matrix = [];
aux = true;
while aux;
    temp = ceil(rand(1,k)*n);
%     temp = temp(1:k);
    matrix = [matrix;temp];
    matrix = unique(matrix,'rows');
    if size(matrix,1)>=stop
       aux=false; 
    end
end

end