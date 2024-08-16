clear all
close all
clc
addpath Functions\
%% Load all data
load data_temp.mat
correc = corre([1 3 5 6]); % correct answer
ncorrec = ncorre([1 3 5 6]); % correct answer normalized
nn=10; % threshold for outlier rejection (units of m.a.d.)
these_ks = [5 10 15 20 25]; % number of individuals on each group
Nk = length(these_ks); % number of iterations
Qs = [1 3 5 6]; % interacted questions


%% Question per question
bias_av=[];
bias_con=[];
dist_av=[];
dist_con=[];
av_i1=[];
av_i2=[];
wgv_i1=[];
wgv_i2=[];

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

bias_av=[bias_av;(mean(ngrp1)-nCA)'];
bias_con=[bias_con;(ndata-nCA)'];
repav  = repmat((mean(ngrp1)),5,1);
repcon = repmat(ndata,5,1);
dist_av=[dist_av; abs(ngrp3(:)-repav(:))];
dist_con=[dist_con;abs(ngrp3(:)-repcon(:))];

wgv_i1=[wgv_i1; var(ngrp1)'];
wgv_i2=[wgv_i2;var(ngrp3)'];


av_i1=[av_i1;(mean(ngrp1))'];
av_i2=[av_i2;(mean(ngrp3))'];

end

%% Figure 2A

mb1 = -mean(bias_av);
mb2 = -mean(bias_con);
eb1 = std(bias_av)/sqrt(length(bias_av));
eb2 = std(bias_con)/sqrt(length(bias_con));

figure('color','w','paperposition',[1 1 5 4]);
bar(1,mb1,.5,'facecolor',.6*[0,0,1],'linewidth',2);hold on
bar(2,mb2,.5,'facecolor',.3*[1,1,1],'linewidth',2)
errorb([1,2],[mb1,mb2],[eb1,eb2],'linewidth',2);
xlim([0.5 2.5]);ylim([0 .45]);box off
set(gca,'xtick',[1:2],'xticklabel',{'average','consensus'},...
    'ytick',[0:.2:.4])
ylabel 'distance from i2 to'


%% Figure 2B

mb1 = mean(dist_av);
mb2 = mean(dist_con);
eb1 = std(bias_av)/sqrt(length(bias_av));
eb2 = std(bias_con)/sqrt(length(bias_con));

figure('color','w','paperposition',[1 1 5 4]);
bar(1,mb1,.5,'facecolor',.6*[0,0,1],'linewidth',2);hold on
bar(2,mb2,.5,'facecolor',.3*[1,1,1],'linewidth',2)
errorb([1,2],[mb1,mb2],[eb1,eb2],'linewidth',2);
xlim([0.5 2.5]);ylim([0 1.3]);box off
set(gca,'xtick',[1:2],'xticklabel',{'average','consensus'},...
    'ytick',[0:.6:1.2])
ylabel 'distance from i2 to'


%% Figure 2C

mb1 = mean(wgv_i1);
mb2 = mean(wgv_i2);
eb1 = std(wgv_i1)/sqrt(length(bias_av));
eb2 = std(wgv_i2)/sqrt(length(bias_con));

figure('color','w','paperposition',[1 1 5 4]);
bar(1,mb1,.5,'facecolor',.6*[0,0,1],'linewidth',2);hold on
bar(2,mb2,.5,'facecolor',.6*[1,0,0],'linewidth',2)
errorb([1,2],[mb1,mb2],[eb1,1.5*eb2],'linewidth',2);
xlim([0.5 2.5]);ylim([0 4.2]);box off
set(gca,'xtick',[1:2],'xticklabel',{'average','consensus'},...
    'ytick',[0:2:4],'yticklabel',{'0.0',[0.2:.2:.4]})
ylabel 'distance from i2 to'



%% Figure 2D

xx=[-4.5:.2:4.5];
[y1,x1]=hist(av_i1,xx);
[y2,x2]=hist(av_i2,xx);
figure('color','w','paperposition',[1 1 6 4]);
plot(x1,y1/nansum(y1),'o','color',.6*[0,0,1],'markerfacecolor',.6*[0,0,1],'markersize',3);hold on
plot(x2,y2/nansum(y2),'o','color',.6*[1,0,0],'markerfacecolor',.6*[1,0,0],'markersize',3);
ind1 = av_i1>-1 & av_i1<1.75;
xx=x1;
ym1=normpdf(xx,mean(av_i1(ind1)),std(av_i1(ind1)));
ym2=normpdf(xx,mean(av_i2),std(av_i2));
hold on;
plot(xx,ym1/sum(ym1),'-','color',.6*[0,0,1],'linewidth',2)
plot(xx,ym2/sum(ym2),'-','color',.6*[1,0,0],'linewidth',2)
xlim([-2.5 4.5]);box off
set(gca,'ytick',[0:0.06:.12],'xtick',[-2:2:4])
xlabel 'bla'
ylabel 'bla'


mb1 = var(av_i1);
mb2 = var(av_i2);

figure('color','w','paperposition',[1 1 5 4]);
bar(1,mb1,.5,'facecolor',.6*[0,0,1],'linewidth',2);hold on
bar(2,mb2,.5,'facecolor',.6*[1,0,0],'linewidth',2)
xlim([0.5 2.5]);ylim([0 2.1]);box off
set(gca,'xtick',[1:2],'xticklabel',{'average','consensus'},...
    'ytick',[0:1:2],'yticklabel',{'0.0',[0.1 .2]})
ylabel 'distance from i2 to'

