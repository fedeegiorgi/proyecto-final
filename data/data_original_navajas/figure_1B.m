clear all
close all
clc

%% Load all data
load data_temp.mat
correc = corre([1 3 5 6]); % correct answer
ncorrec = ncorre([1 3 5 6]); % correct answer normalized
nn=15; % threshold for outlier rejection (units of m.a.d.)
these_ks = [5 10 15 20 25]; % number of individuals on each group
Nk = length(these_ks); % number of iterations
m = 100; % number of samples per iteration
maxrep = 1000; % number of repetitions per sample
Qs = [1 3 5 6]; % interacted questions

%% Pre-define variables
% within error
we_i1=nan(Nk,m,maxrep,4);
we_i2=nan(Nk,m,maxrep,4);
we_j=nan(Nk,m,maxrep,4);
% within error normalized
nwe_i1=nan(Nk,m,maxrep,4);
nwe_i2=nan(Nk,m,maxrep,4);
nwe_j=nan(Nk,m,maxrep,4);
% between error
be_i2=nan(Nk,m,maxrep,4);
% between error normalized
nbe_i2=nan(Nk,m,maxrep,4);
% all groups error
ALL_i1=nan(245,4);
ALL_j=ALL_i1;
ALL_i2=ALL_i1;
% all groups error normalized
nALL_i1=ALL_i1;
nALL_j=ALL_i1;
nALL_i2=ALL_i1;

%% Question per question
for q = 1:4

CA = correc(q); % correct 
nCA = ncorrec(q); % correct normalized

grp1 = squeeze(group_p1(:,q,:));
data = group_p2(q,:);
grp3 = squeeze(group_p3(:,q,:));
ngrp1 = squeeze(ngroup_p1(:,q,:));
ndata = ngroup_p2(q,:);
ngrp3 = squeeze(ngroup_p3(:,q,:));



thr = nanmedian(grp1(:))+nn*mad(grp1(:),1);
iout = grp1>thr;
grp1(iout)=nan;
ngrp1(iout)=nan;
iout = grp3>thr;
grp3(iout)=nan;
ngrp3(iout)=nan;
iout = data>thr;
data(iout)=nan;
ndata(iout)=nan;

out = find(isnan(sum(grp1)) | isnan(sum(grp3)) | isnan(data));

grp1(:,out)=[];data(out)=[];grp3(:,out)=[];
ngrp1(:,out)=[];ndata(out)=[];ngrp3(:,out)=[];

n = size(grp1,2);
NG(q)=n;


for ik=1:Nk
    k=these_ks(ik);
    disp(['n = ',num2str(k) ' - question ',num2str(q)]);
%     disp(num2str(q));
    clear idP flag;
    [idP,flag] = sample_groups(n,k,m);
    for i=1:size(idP,1)
        my_i1 = grp1(:,idP(i,:));
        my_j  = data(idP(i,:));
        my_i2 = grp3(:,idP(i,:));
        my_ni1 = ngrp1(:,idP(i,:));
        my_ni2 = ngrp3(:,idP(i,:));
        my_nj  = ndata(idP(i,:));
        % within group
        gs=nchoosek([1:size(my_i1,2)],ik);
        igs=randsample(size(gs,1),min([maxrep,size(gs,1),]));
        gs=gs(igs,:);
        for ii=1:size(gs,1)
            aux=gs(ii,:);
            wgs_i1=my_i1(:,aux);
            wgs_i2=my_i2(:,aux);
            wgs_j=my_j(aux);
            nwgs_i1=my_ni1(:,aux);
            nwgs_i2=my_ni2(:,aux);
            nwgs_j=my_nj(aux);
            we_i1(ik,i,ii,q) = abs(mean(wgs_i1(:))-CA);
            we_i2(ik,i,ii,q) = abs(mean(wgs_i2(:))-CA);
            we_j(ik,i,ii,q)  = abs(mean(wgs_j(:))-CA);
            nwe_i1(ik,i,ii,q) = abs(mean(nwgs_i1(:))-nCA);
            nwe_i2(ik,i,ii,q) = abs(mean(nwgs_i2(:))-nCA);
            nwe_j(ik,i,ii,q)  = abs(mean(nwgs_j(:))-nCA);
        end
        % between group
        Gs=npermutek2(5,size(my_i1,2),maxrep);
        iGs=randperm(size(Gs,1));
        Gs=Gs(iGs,:);
        for ii=1:size(Gs,1)
            iaux=Gs(ii,:);
            bgs_i2=[];
            nbgs_i2=[];
            for r=1:size(iaux,2)
                bgs_i2=[bgs_i2;my_i2(iaux(r),r)];
                nbgs_i2=[nbgs_i2;my_ni2(iaux(r),r)];
            end
            be_i2(ik,i,ii,q) = abs(mean(bgs_i2(:))-CA);
            nbe_i2(ik,i,ii,q) = abs(mean(nbgs_i2(:))-nCA);
        end
    end
    
end


% ALL datapoint
NN = size(grp1,2);
for i=1:NN
    ALL_i1(i,q)=abs(mean(mean(grp1(:,setdiff([1:nn],i))))-CA);
    ALL_j(i,q)=abs(mean(data(setdiff([1:nn],i)))-CA);
    ALL_i2(i,q)=abs(mean(mean(grp3(:,setdiff([1:nn],i))))-CA);
    nALL_i1(i,q)=abs(mean(mean(ngrp1(:,setdiff([1:nn],i))))-nCA);
    nALL_j(i,q)=abs(mean(ndata(setdiff([1:nn],i)))-nCA);
    nALL_i2(i,q)=abs(mean(mean(ngrp3(:,setdiff([1:nn],i))))-nCA);
end



end

%% Average across questions

mwi1=squeeze(mean(mean(nanmean(nwe_i1,3),2),4));
emwi1=squeeze(std(mean(nanmean(nwe_i1,3),4),[],2))/sqrt(size(nwe_i1,2));
mwj=squeeze(mean(mean(nanmean(nwe_j,3),2),4));
emwj=squeeze(std(mean(nanmean(nwe_j,3),4),[],2))/sqrt(size(nwe_j,2));
mwi2=squeeze(mean(mean(nanmean(nwe_i2,3),2),4));
emwi2=squeeze(std(mean(nanmean(nwe_i2,3),4),[],2))/sqrt(size(nwe_i2,2));
mbi2=squeeze(mean(mean(nanmean(nbe_i2,3),2),4));
embi2=squeeze(std(mean(nanmean(nbe_i2,3),4),[],2))/sqrt(size(nbe_i2,2));


%% Figure 1B

figure('color','w','paperposition',[1 1 7 5]);
plot(these_ks,mwi1,'b-');hold on;
plot(these_ks,mwi2,'r-');
plot(these_ks,mwj,'k-');
errorbar(these_ks,mwi1,emwi1,'b-');hold on;
errorbar(these_ks,mwi2,emwi2,'r-');
errorbar(these_ks,mwj,emwj,'k-');
box off
errorbar(these_ks(end)+5,mean(nanmean(nALL_i1,2)),std(nanmean(nALL_i1,2))/sqrt(size(nALL_i1,1)),'b.');
errorbar(these_ks(end)+5,mean(nanmean(nALL_j,2)),std(nanmean(nALL_j,2))/sqrt(size(nALL_j,1)),'k.');
errorbar(these_ks(end)+5,mean(nanmean(nALL_i2,2)),std(nanmean(nALL_i2,2))/sqrt(size(nALL_i2,1)),'r.');
xlim([3 33])
ylim([.45 1.3])
ylabel 'normalized error'
xlabel '# of individuals (n)'
set(gca,'ytick',[.5 .75 1 1.25],'xtick',[5:5:30],'xticklabel',[5:5:25 1400]);

