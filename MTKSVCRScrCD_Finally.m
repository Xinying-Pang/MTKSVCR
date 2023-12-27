function [OutputSSR]=MTKSVCRScrCD_Finally(x1Train,x1Test,y1Train,y1Test,X1,Y1,x2Train,x2Test,y2Train,y2Test,X2,Y2,x3Train,x3Test,y3Train,y3Test,X3,Y3,x4Train,x4Test,y4Train,y4Test,X4,Y4,x5Train,x5Test,y5Train,y5Test,X5,Y5)
%%% This is a demo of the main function of MTKSVCR with the safe acceleration rule
% Function MTKSVCRScrCD_Finally
% MTKSVCR: A novel multi-task multi-class support vector machine with safe acceleration rule
%   The number of task: T
%   The number of class: K
% Objective of this document:
%         Get the optimal average prediction accuracy and traning time of MTKSVCR(with safe acceleration rule) by using five fold cross validation
%         Get the numer of samples that are screened out
% Input of this document:
%         x1Train--a T*1 cell, each element in the cell represents all training samples from one task; the number 1 in the name x1Train means the first fold
%         y1Train--a T*1 cell, each element in the cell represents the label vector of all training samples from one task
%         x1Test-- aT*1 cell, each element in the cell represents all testing samples from one task
%         y1Test--a T*1 cell, each element in the cell represents the label vector of all testing samples from one task
%         X1--a matrix composed of all training samples in all tasks
%         Y1--a vector composed of the labels of all training samples in all tasks
%     The meanings of  x2Train, x2Test, y2Train, y2Test, X2, Y2, x3Train, x3Test, y3Train, y3Test, X3, Y3, x4Train, x4Test, y4Train, y4Test, X4, Y4, x5Train, x5Test, y5Train, y5Test, X5, Y5 corresponding to different folds are similar which are omitted here
% Output of this document:
%         OutputPrimal--A matrix contains the average prediction accuracy, the training time and the corresponding standard deviation of all parameters
% Run (two steps):
%        1. load('ExampleData.mat')
%        2. MTKSVCRScrCD_Finally(x1Train,x1Test,y1Train,y1Test,X1,Y1,x2Train,x2Test,y2Train,y2Test,X2,Y2,x3Train,x3Test,y3Train,y3Test,X3,Y3,x4Train,x4Test,y4Train,y4Test,X4,Y4,x5Train,x5Test,y5Train,y5Test,X5,Y5)
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
OutputSSR=zeros(num,1);
OutputSSRiden=zeros(num,6);
%% Main code of this document, the loop
s=0;
for i=1:length(delta)
    for j=1:length(p)
        for m=1:length(rho)
            for n=1:length(D)
                for e=1:length(C)
                    s=s+1
                    if e>1
                        D0=D(n);C0=C(e-1);
                    elseif (n>1) & (e==1)
                        D0=D(n-1);C0=C(length(C));
                    elseif (i>1 || j>1 || m>1) & (n==1) & (e==1)
                        D0=D(length(D)); C0=C(length(C));
                    end
                    if s==1
                        B1=cell(T,1);B2=cell(T,1);B3=cell(T,1);B4=cell(T,1);B5=cell(T,1);
                        for rwpre=1:T
                            B1{rwpre,1}=zeros(mtst1(rwpre,1),K);B2{rwpre,1}=zeros(mtst2(rwpre,1),K);B3{rwpre,1}=zeros(mtst3(rwpre,1),K);B4{rwpre,1}=zeros(mtst4(rwpre,1),K);B5{rwpre,1}=zeros(mtst5(rwpre,1),K);
                        end
                        time1all=0;time2all=0;time3all=0;time4all=0;time5all=0;
                        Gamma0All1=cell(K*(K-1)/2,1);Gamma0All2=cell(K*(K-1)/2,1);Gamma0All3=cell(K*(K-1)/2,1);Gamma0All4=cell(K*(K-1)/2,1);Gamma0All5=cell(K*(K-1)/2,1);
                        for xh=1:K*(K-1)/2
                            k1=k1k2(xh,1);
                            k2=k1k2(xh,2);
                            [Bpro_Cell1,time1,gamma01]=MTKSVCR(x1Train,x1Test,y1Train,y1Test,X1,Y1,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 1-fold data
                            [Bpro_Cell2,time2,gamma02]=MTKSVCR(x2Train,x2Test,y2Train,y2Test,X2,Y2,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 2-fold data
                            [Bpro_Cell3,time3,gamma03]=MTKSVCR(x3Train,x3Test,y3Train,y3Test,X3,Y3,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 3-fold data
                            [Bpro_Cell4,time4,gamma04]=MTKSVCR(x4Train,x4Test,y4Train,y4Test,X4,Y4,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 4-fold data
                            [Bpro_Cell5,time5,gamma05]=MTKSVCR(x5Train,x5Test,y5Train,y5Test,X5,Y5,k1,k2,p(j),delta(i),C(e),D(n),rho(m),K);%% call another function MTKSVCR based on the 5-fold data
                            Gamma0All1{xh,1}=gamma01;Gamma0All2{xh,1}=gamma02;Gamma0All3{xh,1}=gamma03;Gamma0All4{xh,1}=gamma04;Gamma0All5{xh,1}=gamma05;
                            for rw=1:T
                                B1{rw,1}=B1{rw,1}+Bpro_Cell1{rw,1};B2{rw,1}=B2{rw,1}+Bpro_Cell2{rw,1};B3{rw,1}=B3{rw,1}+Bpro_Cell3{rw,1};B4{rw,1}=B4{rw,1}+Bpro_Cell4{rw,1};B5{rw,1}=B5{rw,1}+Bpro_Cell5{rw,1};
                            end
                            time1all=time1all+time1;time2all=time2all+time2;time3all=time3all+time3;time4all=time4all+time4;time5all=time5all+time5;
                        end
                        Acc1=zeros(T,1);Acc2=zeros(T,1);Acc3=zeros(T,1);Acc4=zeros(T,1);Acc5=zeros(T,1);
                        for py=1:T
                            [~,Predicty1] = max(B1{py,1},[],2); Acc1(py,1) = mean(Predicty1==y1Test{py,1});%% accuracy of each task of the 1-fold
                            [~,Predicty2] = max(B2{py,1},[],2); Acc2(py,1) = mean(Predicty2==y2Test{py,1});%% accuracy of each task of the 2-fold
                            [~,Predicty3] = max(B3{py,1},[],2); Acc3(py,1) = mean(Predicty3==y3Test{py,1});%% accuracy of each task of the 3-fold
                            [~,Predicty4] = max(B4{py,1},[],2); Acc4(py,1) = mean(Predicty4==y4Test{py,1});%% accuracy of each task of the 4-fold
                            [~,Predicty5] = max(B5{py,1},[],2); Acc5(py,1) = mean(Predicty5==y5Test{py,1});%% accuracy of each task of the 5-fold
                        end
                        meanAcc1=mean(Acc1);meanAcc2=mean(Acc2);meanAcc3=mean(Acc3);meanAcc4=mean(Acc4);meanAcc5=mean(Acc5);
                        meanAcc=[meanAcc1,meanAcc2,meanAcc3,meanAcc4,meanAcc5];
                        OutputSSR(s,1)=mean(meanAcc);OutputSSR(s,2)=100*std(meanAcc,1);
                        time=[time1all,time2all,time3all,time4all,time5all];
                        OutputSSR(s,3)=mean(time);OutputSSR(s,4)=100*std(time,1);
                        OutputSSRiden(s,1)=0;OutputSSRiden(s,2)=0;OutputSSRiden(s,3)=0;OutputSSRiden(s,4)=0;OutputSSRiden(s,5)=0;OutputSSRiden(s,6)=0;
                    else
                        B1=cell(T,1);B2=cell(T,1);B3=cell(T,1);B4=cell(T,1);B5=cell(T,1);
                        for rwpre=1:T
                            B1{rwpre,1}=zeros(mtst1(rwpre,1),K);B2{rwpre,1}=zeros(mtst2(rwpre,1),K);B3{rwpre,1}=zeros(mtst3(rwpre,1),K);B4{rwpre,1}=zeros(mtst4(rwpre,1),K);B5{rwpre,1}=zeros(mtst5(rwpre,1),K);
                        end
                        time1all=0;time2all=0;time3all=0;time4all=0;time5all=0;
                        for xh=1:K*(K-1)/2
                            k1=k1k2(xh,1);
                            k2=k1k2(xh,2);
                            [Bpro_cell1,time1,id01,id11,All01,All11,Per01,Per11,gamma01]=MTKSVCRScrCD(x1Train,x1Test,y1Train,y1Test,X1,Y1,k1,k2,p(j),delta(i),C(e),D(n),C0,D0,rho(m),K,Gamma0All1{xh,1});%% call another function MTKSVCRScrCD based on the 1-fold data
                            [Bpro_cell2,time2,id02,id12,All02,All12,Per02,Per12,gamma02]=MTKSVCRScrCD(x2Train,x2Test,y2Train,y2Test,X2,Y2,k1,k2,p(j),delta(i),C(e),D(n),C0,D0,rho(m),K,Gamma0All2{xh,1});%% call another function MTKSVCRScrCD based on the 2-fold data
                            [Bpro_cell3,time3,id03,id13,All03,All13,Per03,Per13,gamma03]=MTKSVCRScrCD(x3Train,x3Test,y3Train,y3Test,X3,Y3,k1,k2,p(j),delta(i),C(e),D(n),C0,D0,rho(m),K,Gamma0All3{xh,1});%% call another function MTKSVCRScrCD based on the 3-fold data
                            [Bpro_cell4,time4,id04,id14,All04,All14,Per04,Per14,gamma04]=MTKSVCRScrCD(x4Train,x4Test,y4Train,y4Test,X4,Y4,k1,k2,p(j),delta(i),C(e),D(n),C0,D0,rho(m),K,Gamma0All4{xh,1});%% call another function MTKSVCRScrCD based on the 4-fold data
                            [Bpro_cell5,time5,id05,id15,All05,All15,Per05,Per15,gamma05]=MTKSVCRScrCD(x5Train,x5Test,y5Train,y5Test,X5,Y5,k1,k2,p(j),delta(i),C(e),D(n),C0,D0,rho(m),K,Gamma0All5{xh,1});%% call another function MTKSVCRScrCD based on the 5-fold data
                            Gamma0All1{xh,1}=gamma01;Gamma0All2{xh,1}=gamma02;Gamma0All3{xh,1}=gamma03;Gamma0All4{xh,1}=gamma04;Gamma0All5{xh,1}=gamma05;
                            for rw=1:T
                                B1{rw,1}=B1{rw,1}+Bpro_cell1{rw,1};B2{rw,1}=B2{rw,1}+Bpro_cell2{rw,1};B3{rw,1}=B3{rw,1}+Bpro_cell3{rw,1};B4{rw,1}=B4{rw,1}+Bpro_cell4{rw,1};B5{rw,1}=B5{rw,1}+Bpro_cell5{rw,1};
                            end
                            time1all=time1all+time1;time2all=time2all+time2;time3all=time3all+time3;time4all=time4all+time4;time5all=time5all+time5;
                        end
                        Acc1=zeros(T,1);Acc2=zeros(T,1);Acc3=zeros(T,1);Acc4=zeros(T,1);Acc5=zeros(T,1);
                        for py=1:T
                            [~,Predicty1] = max(B1{py,1},[],2); Acc1(py,1) = mean(Predicty1==y1Test{py,1});%% accuracy of each task of the 1-fold
                            [~,Predicty2] = max(B2{py,1},[],2); Acc2(py,1) = mean(Predicty2==y2Test{py,1});%% accuracy of each task of the 2-fold
                            [~,Predicty3] = max(B3{py,1},[],2); Acc3(py,1) = mean(Predicty3==y3Test{py,1});%% accuracy of each task of the 3-fold
                            [~,Predicty4] = max(B4{py,1},[],2); Acc4(py,1) = mean(Predicty4==y4Test{py,1});%% accuracy of each task of the 4-fold
                            [~,Predicty5] = max(B5{py,1},[],2); Acc5(py,1) = mean(Predicty5==y5Test{py,1});%% accuracy of each task of the 5-fold
                        end
                        meanAcc1=mean(Acc1);meanAcc2=mean(Acc2);meanAcc3=mean(Acc3);meanAcc4=mean(Acc4);meanAcc5=mean(Acc5);
                        meanAcc=[meanAcc1,meanAcc2,meanAcc3,meanAcc4,meanAcc5];
                        OutputSSR(s,1)=mean(meanAcc);%% the average accuracy
                        OutputSSR(s,2)=100*std(meanAcc,1);%% the standard deviation of accuracy
                        time=[time1all,time2all,time3all,time4all,time5all]; OutputSSR(s,3)=mean(time);%% the average training time
                        OutputSSR(s,4)=100*std(time,1);%% the standard deviation of training time
                        id0All=[id01,id02,id03,id04,id05];id1All=[id11,id12,id13,id14,id15];
                        All0All=[All01,All02,All03,All04,All05];All1All=[All11,All12,All13,All14,All15];
                        Per0All=[Per01,Per02,Per03,Per04,Per05]; Per1All=[Per11,Per12,Per13,Per14,Per15];
                        OutputSSRiden(s,1)=mean(id0All);OutputSSRiden(s,2)=mean(id1All);OutputSSRiden(s,3)=mean(All0All);OutputSSRiden(s,4)=mean(All1All);
                        OutputSSRiden(s,5)=mean(Per0All);OutputSSRiden(s,6)=mean(Per1All);
                    end
                    save SACDMTKSVCR_result OutputSSR OutputSSRiden %% save the result as two matrices termed as OutputSSR and OutputSSRiden
                end
            end
        end
    end
end