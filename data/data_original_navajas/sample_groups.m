function [idP,flag] = sample_groups(Ngroups,Ksamps,Mreps)

n = Ngroups;
k = Ksamps;
m = Mreps;
flag=0;
idP = [];
count = 0;
if Ksamps>1
    while size(idP,1)<m
        in=randperm(n);
        idP = [idP;in(1:k)];
        idP = unique(idP,'rows');
        if count>2*m
            break
        end
    end
elseif Ksamps==1
    l=min(Mreps,Ngroups);
    in=randperm(n)';
    idP = in(1:l);
else
    error('Ksamps must be equal or greater than 1')
end

if size(idP,1)<m
    flag=1;
end


end