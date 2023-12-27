function [Bpro_cell,time,id0,id1,All0,All1,Per0,Per1,gamma]=MTKSVCRScrCD(xTrain,xTest,yTrain,yTest,X,Y,k1,k2,p,delta,cc,dd,c0,d0,rho,class,gamma0)
%%% This is the demo of the function corresponding to the inner loop(with safe acceleration rule)
% Function MTKSVCRScrCD
% MTKSVCR: A novel multi-task multi-class support vector machine with safe acceleration rule
%   The number of task: T
%   l_t--The number of samples in the t-th task; l_t^12--The number of samples of two focused classes
% Primal Problem:
%         min  \frac{1}{2}\|w_0\|^2+\sum_{t=1}^{T}\frac{rho}{2}\|w_t\|^2+cc\sum_{t=1}^{T}\sum_{i=1}^{l_{t}^{12}}\xi_{it}+dd\sum_{t=1}^{T}\sum_{m=l_{t,12}+1}^{l_t}(\varphi_{mt}+\varphi_{mt}^{*})
%         s.t.    y_{it}(w_0+w_t)^{T}\phi(x_{it})\geq 1-\xi_{it}
%                 -delta-\varphi_{mt}^{*} \leq (w_0+w_t)^{T}\phi({x_{mt})\leq delta + \varphi_{mt}
%                 \xi_{it}\geq 0, \varphi_{mt}\geq 0, \varphi_{mt}^{*}\geq 0
%  Objective of this document: Get the prediction value, training time, the numer of the screened samples, the percentage of the screened samples and the optimal dual solution corresponding to each parameter
%  Input of this document:
%       xTrain--A T*1 cell, each element in the cell represents all training samples from one task
%       xTest-- A T*1 cell, each element in the cell represents all testing samples from one task
%       yTrain--A T*1 cell, each element in the cell represents the label vector of all training samples from one task
%       yTest--A T*1 cell, each element in the cell represents the label vector of all testing samples from one task
%       X--A matrix composed of all training samples in all tasks
%       Y--A vector composed of the labels of all training samples in all tasks
%       k1,k2--Label of current classifier, eg. k1=1,k2=2(select the first and the second classes of samples as the focused samples); k1=1, k2=3; k1=2, k2=3
%       class--The number of classes
%       gamma0--The optimal dual solution with parameters (c0,d0)
% Parameters:
%       p--The kernel parameter of Gaussian kernel function
%       delta--An artificial parameter in the constraint which should be assigned
%       cc--Penalty paramter in the objective function
%       dd--Penalty parameter in the objective function
%       c0--Penalty parameter in the objective function, similar with cc, but the values of cc and c0 are different
%       d0--Penalty parameter in the objective function, similar with dd, but the values of dd and d0 are different
%       rho--Penalty parameter in the objective function
% Output of this document:
%       Bpro_cell--The prediction matrix, the elements are either 0 or 1
%       id0, id1--The number of samples that are screened out
%       All0, All1--The number of 0 or constant elements in the dual optimal solution
%       Per0--id0/All0;Per1--id1/All1
%       time--The training time of this document
%       gamma--The obtained optimal dual solution corresponding to a specific parameter
%% Preparation
solver = struct('Display', 'off');
T=size(xTrain,1);
XIJ=cell(T,1);XR=cell(T,1);
y=cell(T,1);
Lp=zeros(T,1);Ln=zeros(T,1);Lpn=zeros(T,1);L3=zeros(T,1);mtst=zeros(T,1);
Xmid=xTrain;
for i=1:T
    Indexp=find(yTrain{i,1}==k1);
    Indexn=find(yTrain{i,1}==k2);
    XIJ{i,1}=[xTrain{i,1}(Indexp,:);xTrain{i,1}(Indexn,:)];
    Xmid{i,1}([Indexp;Indexn],:)=[];
    XR{i,1}=Xmid{i,1};
    Lp(i,1)=nnz(yTrain{i,1}==k1);%% the number of positive samples
    Ln(i,1)=nnz(yTrain{i,1}==k2);%% the number of negative samples
    Lpn(i,1)=Lp(i,1)+Ln(i,1);
    y{i,1}=[ones(Lp(i,1),1);-ones(Ln(i,1),1)];
    L3(i,1)=size(yTrain{i,1},1)-Lpn(i,1);%%  the number of remaining samples
    mtst(i,1)=size(xTest{i,1},1);
end
LpnAll=sum(Lpn);L3All=sum(L3);
L=LpnAll+L3All+L3All;
%% Construct the kernel matrix of training samples
Ker1=cell(T,T);%% kernel matrix of the focused classes of samples
Ker2=cell(T,T);%% kernel matrix of the focused classes of samples and the remaining classes of samples
Ker3=cell(T,T);%% kernel matrix of the remaining classes of samples
for i=1:T
    for j=1:T
        Ker1{i,j}=exp(-(repmat(sum(XIJ{i,1}.*XIJ{i,1},2),1,Lpn(j,1))+repmat(sum(XIJ{j,1}.*XIJ{j,1},2)',Lpn(i,1),1) - 2*XIJ{i,1}*XIJ{j,1}')/(2*p^2));
        Ker2{i,j}=exp(-(repmat(sum(XIJ{i,1}.*XIJ{i,1},2),1,L3(j,1))+repmat(sum(XR{j,1}.*XR{j,1},2)',Lpn(i,1),1) - 2*XIJ{i,1}*XR{j,1}')/(2*p^2));
        Ker3{i,j}=exp(-(repmat(sum(XR{i,1}.*XR{i,1},2),1,L3(j,1))+repmat(sum(XR{j,1}.*XR{j,1},2)',L3(i,1),1) - 2*XR{i,1}*XR{j,1}')/(2*p^2));
    end
end
Yall=diag(cell2mat(y));
K1=cell2mat(Ker1);%% all focused samples*all focused samples
K2=cell2mat(Ker2);%% all focused samples*all remaining samples
K3=cell2mat(Ker3);%% all remaining samples*all remaining samples

Re=reshape(1:T*T,T,T);
ReIndex=diag(Re);
DK1_pre=Ker1(ReIndex);
DK1=blkdiag(DK1_pre{:});
DK2_pre=Ker2(ReIndex);
DK2=blkdiag(DK2_pre{:});
DK3_pre=Ker3(ReIndex);
DK3=blkdiag(DK3_pre{:});
H1=Yall*K1*Yall+(1/rho)*Yall*DK1*Yall;
H2=-Yall*K2-(1/rho)*Yall*DK2;
H3=K3+(1/rho)*DK3;
H=[H1 H2 -H2; H2' H3 -H3; -H2' -H3' H3'];
H=(H+H')/2;
f=[-ones(LpnAll,1);delta*ones(L3All,1);delta*ones(L3All,1)];
A=[];b=[];
Aeq=[];beq=[];
ub=[cc*ones(LpnAll,1);dd*ones(L3All,1);dd*ones(L3All,1)];
%% Screening before solving
gamma_end=-ones(L,1);
BP=diag([((cc+c0)/(2*c0))*ones(LpnAll,1);((dd+d0)/(2*d0))*ones(L3All,1);((dd+d0)/(2*d0))*ones(L3All,1)]);
BM=diag([((cc-c0)/(2*c0))*ones(LpnAll,1);((dd-d0)/(2*d0))*ones(L3All,1);((dd-d0)/(2*d0))*ones(L3All,1)]);
tic;
var1=sqrt(diag(H))*sqrt(gamma0'*BM*H*BM*gamma0);
var2=H*BP*gamma0;
Q=var2-var1>-f+(1e-8);
F=var2+var1<-f-(1e-8);
gamma_end(Q==1)=0;
id0=nnz(Q);
gamma_end(F==1)=ub(F==1);
id1=nnz(F);
idx_delete=(gamma_end~=-1);
reSam=nnz(idx_delete==0);
gamma_t=gamma_end;
gamma_t(gamma_t==-1)=[];
ub_re=ub(idx_delete==0);
f_re=f(idx_delete==0)+H(idx_delete==0,idx_delete==1)*gamma_t;
H_re=H(idx_delete==0,idx_delete==0);
%% The solving process
if reSam~=0
    lb=zeros(reSam,1);
    [gamma_re]=quadprog(H_re,f_re,A,b,Aeq,beq,lb,ub_re,[],solver);
    gamma_end(idx_delete==0)=gamma_re;
end
time=toc;
gamma=gamma_end;
All0=nnz(gamma_end<(1e-8));
All1=nnz(gamma_end>(ub-(1e-8)));
if All0==0
    Per0=0;
else
    Per0=id0/All0;
end
if All1==0
    Per1=0;
else
    Per1=id1/All1;
end
%% Record the solution of each task
ha_m=0;ha_later=0;Alpha=cell(T,1);%% alpha
hb_m=LpnAll;hb_later=LpnAll;Beta=cell(T,1); %% beta
hbs_m=LpnAll+L3All;hbs_later=LpnAll+L3All;BetaS=cell(T,1);%% beta*
for i=1:T
    ha_before=ha_m+1; hb_before=hb_m+1; hbs_before=hbs_m+1;
    ha_later=ha_later+Lpn(i,1); hb_later=hb_later+L3(i,1); hbs_later=hbs_later+L3(i,1);
    ha_m=ha_later; hb_m=hb_later; hbs_m=hbs_later;
    Alpha{i,1}=gamma(ha_before:ha_later,1);
    Beta{i,1}=gamma(hb_before:hb_later,1);
    BetaS{i,1}=gamma(hbs_before:hbs_later,1);
end
%% Construct the kernel matrix of testing samples
tstKer1=cell(T,T);
for i=1:T
    for j=1:T
        tstKer1{i,j}=exp(-(repmat(sum(xTest{i,1}.*xTest{i,1},2),1,Lpn(j,1))+repmat(sum(XIJ{j,1}.*XIJ{j,1},2)',mtst(i,1),1) - 2*xTest{i,1}*XIJ{j,1}')/(2*p^2));
    end
end
tstKer2=cell(T,T);
for i=1:T
    for j=1:T
        tstKer2{i,j}=exp(-(repmat(sum(xTest{i,1}.*xTest{i,1},2),1,L3(j,1))+repmat(sum(XR{j,1}.*XR{j,1},2)',mtst(i,1),1) - 2*xTest{i,1}*XR{j,1}')/(2*p^2));
    end
end
%% The testing process
w0x=cell(T,1);wtx=cell(T,1);
PFunVal=cell(T,1);
for i=1:T
    tstKer1use=cell2mat(tstKer1(i,:));
    tstKer2use=cell2mat(tstKer2(i,:));
    w0x{i,1}=tstKer1use*(Yall*gamma(1:LpnAll,1))-tstKer2use*(gamma((LpnAll+1):(LpnAll+L3All),1)-gamma((LpnAll+L3All+1):end,1));
    wtx{i,1}=tstKer1{i,i}*(diag(y{i,1})*Alpha{i,1})-tstKer2{i,i}*(Beta{i,1}-BetaS{i,1});
    PFunVal{i,1}=w0x{i,1}+(1/rho)*wtx{i,1};
end
%% Get the prediction matrix
Bpro_cell=cell(T,1);
for i=1:T
    B_pro=zeros(mtst(i,1),class);
    Index1=(PFunVal{i,1}>=delta); %% first class
    Index2=(PFunVal{i,1}<=(-delta)); %% second class
    Index3=(PFunVal{i,1}>-delta & PFunVal{i,1} <delta); %% other classes
    B_pro(Index1,k1)=1; %% assign values based on predicted results
    B_pro(Index2,k2)=1;
    B_pro(Index3,[k1 k2])=-1;
    Bpro_cell{i,1}=B_pro;
end