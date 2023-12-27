function [OutputPrimal]=MTKSVCR_Finally(x1Train,x1Test,y1Train,y1Test,X1,Y1,x2Train,x2Test,y2Train,y2Test,X2,Y2,x3Train,x3Test,y3Train,y3Test,X3,Y3,x4Train,x4Test,y4Train,y4Test,X4,Y4,x5Train,x5Test,y5Train,y5Test,X5,Y5)
%%% This is a demo of the main function of MTKSVCR
% Function MTKSVCR_Finally:
% MTKSVCR: A novel multi-task multi-class support vector machine with safe acceleration rule
%   The number of task: T
%   The number of class: K
% Objective of this document: Get the optimal average prediction accuracy and traning time of MTKSVCR by using five fold cross validation
% Input of this document:
%     x1Train--a T*1 cell, each element in the cell represents all training samples from one task; the number 1 in the name x1Train means the first fold
%     y1Train--a T*1 cell, each element in the cell represents the label vector of all training samples from one task
%     x1Test-- aT*1 cell, each element in the cell represents all testing samples from one task
%     y1Test--a T*1 cell, each element in the cell represents the label vector of all testing samples from one task
%     X1--a matrix composed of all training samples in all tasks
%    Y1--a vector composed of the labels of all training samples in all tasks
%     The meanings of  x2Train, x2Test, y2Train, y2Test, X2, Y2, x3Train, x3Test, y3Train, y3Test, X3, Y3, x4Train, x4Test, y4Train, y4Test, X4, Y4, x5Train, x5Test, y5Train, y5Test, X5, Y5 corresponding to different folds are similar which are omitted here
% Output of this document:
%    OutputPrimal--A matrix contains the average prediction accuracy, the training time and the corresponding standard deviation of all parameters
% Run (two steps):
% 1. load('ExampleData.mat')
% 2. MTKSVCR_Finally(x1Train,x1Test,y1Train,y1Test,X1,Y1,x2Train,x2Test,y2Train,y2Test,X2,Y2,x3Train,x3Test,y3Train,y3Test,X3,Y3,x4Train,x4Test,y4Train,y4Test,X4,Y4,x5Train,x5Test,y5Train,y5Test,X5,Y5)
%% Set the range of all parameters
D=2.^[1:0.01:2];
C=2.^[1:0.01:2];
p=8;
rho=8;
delta=0.01;
%% Preparation
T=size(x1Train,1);
K=max(y1Train{1,1});
k1k2 = [];
for k1 = 1:K-1
    for k2 = k1+1:K
        k1k2 = [k1k2;k1 k2];
    end
end
mtst1=zeros(T,1);mtst2=zeros(T,1);mtst3=zeros(T,1);mtst4=zeros(T,1);mtst5=zeros(T,1);
for i=1:T
    mtst1(i,1)=size(x1Test{i,1},1);mtst2(i,1)=size(x2Test{i,1},1);mtst3(i,1)=size(x3Test{i,1},1);mtst4(i,1)=size(x4Test{i,1},1);mtst5(i,1)=size(x5Test{i,1},1);
end
num=length(C)*length(D)*length(delta)*length(p)*length(rho);
OutputPrimal=zeros(num,1);

%% Main code of this document, the loop
s=0;
for i=1:length(delta)
    for j=1:length(p)
        for m=1:length(rho)
            for n=1:length(D)
                for e=1:length(C)
                    s=s+1
                    B1=cell(T,1);B2=cell(T,1);B3=cell(T,1);B4=cell(T,1);B5=cell(T,1);
                    for rwpre=1:T
                        B1{rwpre,1}=zeros(mtst1(rwpre,1),K);B2{rwpre,1}=zeros(mtst2(rwpre,1),K);B3{rwpre,1}=zeros(mtst3(rwpre,1),K);B4{rwpre,1}=zeros(mtst4(rwpre,1),K);B5{rwpre,1}=zeros(mtst5(rwpre,1),K);
                    end
                    time1all=0;time2all=0;time3all=0;time4all=0;time5all=0;
                    for xh=1:K*(K-1)/2
                        k1=k1k2(xh,1);
                        k2=k1k2(xh,2);
                        [Bpro_Cell1,time1,~]=MTKSVCR(x1Train,x1Test,y1Train,y1Test,X1,Y1,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 1-fold data
                        [Bpro_Cell2,time2,~]=MTKSVCR(x2Train,x2Test,y2Train,y2Test,X2,Y2,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 2-fold data
                        [Bpro_Cell3,time3,~]=MTKSVCR(x3Train,x3Test,y3Train,y3Test,X3,Y3,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K); %% call another function MTKSVCR based on the 3-fold data
                        [Bpro_Cell4,time4,~]=MTKSVCR(x4Train,x4Test,y4Train,y4Test,X4,Y4,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 4-fold data
                        [Bpro_Cell5,time5,~]=MTKSVCR(x5Train,x5Test,y5Train,y5Test,X5,Y5,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 5-fold data
                        for rw=1:T
                            B1{rw,1}=B1{rw,1}+Bpro_Cell1{rw,1};B2{rw,1}=B2{rw,1}+Bpro_Cell2{rw,1};B3{rw,1}=B3{rw,1}+Bpro_Cell3{rw,1};B4{rw,1}=B4{rw,1}+Bpro_Cell4{rw,1};B5{rw,1}=B5{rw,1}+Bpro_Cell5{rw,1};
                        end
                        time1all=time1all+time1;time2all=time2all+time2;time3all=time3all+time3;time4all=time4all+time4;time5all=time5all+time5;
                    end
                    Acc1=zeros(T,1);Acc2=zeros(T,1);Acc3=zeros(T,1);Acc4=zeros(T,1);Acc5=zeros(T,1);
                    for py=1:T
                        [~,Predicty1] = max(B1{py,1},[],2); Acc1(py,1) = mean(Predicty1==y1Test{py,1}); %% accuracy of each task of the 1-fold
                        [~,Predicty2] = max(B2{py,1},[],2); Acc2(py,1) = mean(Predicty2==y2Test{py,1}); %% accuracy of each task of the 2-fold
                        [~,Predicty3] = max(B3{py,1},[],2); Acc3(py,1) = mean(Predicty3==y3Test{py,1}); %% accuracy of each task of the 3-fold
                        [~,Predicty4] = max(B4{py,1},[],2); Acc4(py,1) = mean(Predicty4==y4Test{py,1}); %% accuracy of each task of the 4-fold
                        [~,Predicty5] = max(B5{py,1},[],2); Acc5(py,1) = mean(Predicty5==y5Test{py,1}); %% accuracy of each task of the 5-fold
                    end
                    meanAcc1=mean(Acc1);meanAcc2=mean(Acc2);meanAcc3=mean(Acc3);meanAcc4=mean(Acc4);meanAcc5=mean(Acc5);
                    meanAcc=[meanAcc1,meanAcc2,meanAcc3,meanAcc4,meanAcc5];
                    OutputPrimal(s,1)=mean(meanAcc);%% the average accuracy
                    OutputPrimal(s,2)=100*std(meanAcc,1); %% the standard deviation of accuracy
                    time=[time1all,time2all,time3all,time4all,time5all];  OutputPrimal(s,3)=mean(time);%% the average training time
                    OutputPrimal(s,4)=100*std(time,1); %% the standard deviation of training time
                    save MTKSVCR_result OutputPrimal %% save the result as a matrix termed as OutputPrimal
                end
            end
        end
    end
end