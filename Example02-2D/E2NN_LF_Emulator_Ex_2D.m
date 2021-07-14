function E2NN_Ex_2D_v02
% Runs E2NN while using a Low-Fidelity model as an emulator

% NOTE: accuracy of the E2NN and other fits may vary from run to run
% because no random seed was set.

close all
clear; clc;

%% Training data

% Sn is the number of training samples
Sn=12;%24
nd=2;
vali_sample=0
min_SE_target=0.0000001
max_epoch=30
restart_epoch=4

% Loading specific training data for more consistent results
if Sn==12
    load temp_result_goodresults01_SN12_0251.mat
elseif Sn==24
    load temp_result_goodresults01_SN24_0064.mat
elseif Sn==30
    load temp_result_goodresults01_SN30_0047.mat
elseif Sn==40
    load temp_result_goodresults01_SN40_0024.mat
elseif Sn==60
    load temp_result_goodresults01_SN60_0013.mat
elseif Sn==100
    load temp_result_goodresults01_SN100_0009.mat
else
%     Xsn=rand(Sn,nd);
    Xsn=lhsdesign(Sn-2,nd);
    Xsn=[Xsn;0.87 0.98;0.02 0.95]*1.00;
    Sn=length(Xsn(:,1));
    %%%%% ==> Test function 
    Ysn=test_function_2D(Xsn);
    Ystats.mean=mean(Ysn);
    Ystats.std=std(Ysn);
    Ystats.max=max(Ysn);
    Ystats.min=min(Ysn);
    Ysn=(Ysn-Ystats.mean)/Ystats.std;
end

Lamda=0.001

%% NN model definition

NNs=[nd 20 20 1]; %number of neurons at each layer
LP=length(NNs);% number of layers
%
%  Ws: (Np+1 by Np:  output neuron number by input neuron number
%  Bs:  Np+1 by 1 :  output neuron number by one
%
%% Data Mini-batch setup
Bn=1000;
NB=floor(Sn/Bn);
Bns=[1:NB-1]*Bn;
Bns=[1 Bns Sn-vali_sample];
NB=length(Bns)-1;

% SnT=Sn;
% YsnT=Ysn;
% XsnT=Xsn;

%% Initializing the Weights
for p=1:LP-1
    fan_ave=(NNs(p+1)+NNs(p))/2;
    stddev=sqrt(1/fan_ave);
    Ws{p}=randn(NNs(p+1),NNs(p))*stddev+eps;
    Bs{p}=zeros(NNs(p+1),1);
end

%%%% variable flattening to Wsdl
Wsdl=[];
for p=1:LP-1
    Wsi=Ws{p};
    Bsi=Bs{p};
    Wsdl=[Wsdl;Wsi(:);Bsi(:)];
end


%% NN mini-batch normalized fitting
global opti_history
opti_history.x=[];
opti_history.fval=[];

min_SE=inf;
best_Wsdl=Wsdl;
best_epoch=0;

SE_history=dlarray(zeros(1,1));
options = optimoptions('fminunc','SpecifyObjectiveGradient',true,...
     'OutputFcn', @opti_iter_output,...
     'StepTolerance',0.0001,...
     'OptimalityTolerance',0.0001,...
     'MaxIterations',100,...
     'display','off');
%      'Algorithm','trust-region',...

epoch=1;

while (epoch<(max_epoch+1) && min_SE>min_SE_target)
    for bi=1:NB
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Batch 01
        Ysbi=Ysn(Bns(bi):Bns(bi+1));
        Xsbi=Xsn(Bns(bi):Bns(bi+1),:);
        
        
        Wsdl_opt=fminunc(@net_train_costobj_autograd,Wsdl,options,Xsn,Ysn,Ystats,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda);
        Wsdl=Wsdl_opt;

        [Ws, Bs]=reshaping_Ws_Bs(Wsdl,Ws,Bs);
        [Yp,as,zs]=feedforward(Xsn,LP,NNs,Ws,Bs,Xsn,Ysn,Ystats);  

         % normalized RMSE (NRMSE)
         SE=sqrt(sum((Yp-Ysn).^2)/length(Ysn));
         NRMSE=SE/(max(Ysn)-min(Ysn));     



         SSw=0;    
         for i=1:length(Ws)
             SSw=SSw+sum(sum(Ws{i}.^2));
         end

         cost=NRMSE+Lamda*SSw;       

        SE_history(epoch)=cost;
        opti_history.fval=[opti_history.fval cost];
         
        if min_SE>cost
           min_SE=cost;
           best_Wsdl=Wsdl; 
           best_epoch=epoch;
        end
        
        
        temp_str=sprintf('Epoch: %d (%d)  Restarted: %d  %f %f cost: %f  min_cost: %f',epoch, bi, floor(epoch/restart_epoch), NRMSE,  SSw, cost,min_SE);
        disp(temp_str)
        
    end
    
    new_order=randperm(Sn);
    Xsn=Xsn(new_order,:);
    Ysn=Ysn(new_order);

    if mod(epoch,restart_epoch)==0 & min_SE>min_SE_target
        %%%% Reinitialization of Weights
        Wsdl=[];
        for p=1:LP-1
            fan_ave=(NNs(p+1)+NNs(p))/2;
            stddev=sqrt(1/fan_ave);
            Ws{p}=randn(NNs(p+1),NNs(p))*stddev*2+eps;
            Bs{p}=zeros(NNs(p+1),1);
            Wsdl=[Wsdl;Ws{p}(:);Bs{p}(:)];
        end
    end
    epoch=epoch+1;
end
        
       
[Ws, Bs]=reshaping_Ws_Bs(best_Wsdl,Ws,Bs);


%% Matlab ML function
% net=feedforwardnet(NNs(2:end-1));
% % net.divideParam.trainRatio=0.8;
% % net.divideParam.valRatio=0.2;
% % net.divideParam.testRatio=0;
% net.trainFcn = 'trainbr';
% net.trainParam
% 
% 
% [net,tr]=train(net, Xsn', Ysn'); %% input_x should be the size of nd-by-Sn



options = trainingOptions('sgdm','L2Regularization', Lamda,... %1e-3); % rmsprop %adam
                          'MaxEpochs',20000, ...
                          'MiniBatchSize',128,...'LearnRateDropFactor', 0.0,...
                          'Plots', 'training-progress',...
                          'BatchNormalizationStatistics', 'population');
     
layers = [ ...
    featureInputLayer(2)
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(10)
    tanhLayer
    fullyConnectedLayer(1)
    regressionLayer];



net = trainNetwork(Xsn, Ysn, layers, options);






% % load temp_result_goodresults01_SN12_0251.mat

%% NN result visualization

% iteration history
figure, semilogy(opti_history.fval)

% 3D plots

grid=linspace(0,1,30)
[xs1,xs2]=meshgrid(grid,grid);
shape=size(xs1);
xs1=xs1(:);
xs2=xs2(:);

[Yp_bh,as,zs]=feedforward([xs1 xs2],LP,NNs,Ws,Bs,Xsn,Ysn,Ystats);
% Yp_net=net([xs1 xs2]');
Yp_net=predict(net, [xs1 xs2]);
Ys=test_function_2D([xs1 xs2]);
Ys_LF01=test_function_2D_LF01([xs1 xs2]);

%%%% ==> De-normalization (back-normalization)
% Yp_bh=(Ystd)*Yp_bh+Ymean;
% Yp_net=(Ystd)*Yp_net+Ymean;
% Ysn=(Ystd)*Ysn+Ymean;

Yp_bh=Yp_bh*Ystats.std+Ystats.mean;
Yp_net=Yp_net*Ystats.std+Ystats.mean;
Ysn=Ysn*Ystats.std+Ystats.mean;

n=length(Ys)
div=Ys-Yp_bh;
RMSE_e2nn=sqrt(sum(div.^2)/n)
[maxdiv, id]=max(abs(div))
xs1(id)
xs2(id)


n=length(Ys)
div=Ys-Yp_net(:);
RMSE_net=sqrt(sum(div.^2)/n)
[maxdiv, id]=max(abs(div))
xs1(id)
xs2(id)

[RMSE_e2nn RMSE_net]



xs1=reshape(xs1,shape);
xs2=reshape(xs2,shape);
Ys=reshape(Ys,shape);
Ys_LF01=reshape(Ys_LF01,shape);
Yp_bh=reshape(Yp_bh,shape);
Yp_net=reshape(Yp_net,shape);

% plot ordinary LF model
figure, hold on, view([-25 20]), rotate3d
surface(xs1,xs2,Ys,'FaceAlpha',0.2);
surface(xs1,xs2,Ys_LF01,'FaceColor','g', 'FaceAlpha',0.6)
scatter3(Xsn(:,1),Xsn(:,2),Ysn,80,'filled',...
            'MarkerEdgeColor','k',...
        'MarkerFaceColor','r')
% view([120,30]);
% legend('HF', 'LF')

% plot ordinary NN
figure, hold on, view([-25 20]), rotate3d
surface(xs1,xs2,Ys);
surface(xs1,xs2,Yp_net,'FaceColor','m')
scatter3(Xsn(:,1),Xsn(:,2),Ysn,80,'filled',...
            'MarkerEdgeColor','k',...
        'MarkerFaceColor','r')
% view([120,30]);
% legend('HF', 'Ordinary NN')

% plot E2NN
figure, hold on, view([-25 20]), rotate3d
surface(xs1,xs2,Ys);
surface(xs1,xs2,Yp_bh,'FaceColor','b');
scatter3(Xsn(:,1),Xsn(:,2),Ysn,80,'filled',...
            'MarkerEdgeColor','k',...
        'MarkerFaceColor','r')
% view([120,30]);
% legend('HF', 'E2NN')


RMSE_e2nn 
RMSE_net
Ws{1};
Ws{2};
Bs{1};
Bs{2};

% 
% sampleN=5000;
% xs=lhsdesign(sampleN,2);
% [Ys,sy,ry]=test_function_2D([xs(:,1) xs(:,2)]);
% scatter3(xs(:,1), xs(:,2), ry, 'r.')
% 



opti_history;


end
 

%% Imbedded function for opti history
function stop = opti_iter_output(x,optimValues,state,Xsn,Ysn,Ystats,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda)

global opti_history

  stop = false;
  if isequal(state,'iter')
    opti_history.x = [opti_history.x  x];
    opti_history.fval = [opti_history.fval  optimValues.fval];
  end
end
     
     
function [cost, dcost]=net_train_costobj_autograd(Wsdl,Xsn,Ysn,Ystats,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda)
    
%      Wsdl_opt=fminunc(@net_train_opt_autograd,Wsdl,options,Ysn,Xsn,LP,NNs,Ws,Bs); 
    Wsdl=dlarray(Wsdl);
    [cost, dcost]=dlfeval(@LSE_costfun,Wsdl,Xsn,Ysn,Ystats,Xsbi,Ysbi,LP,NNs,Ws,Bs,Lamda);
    cost=extractdata(cost);
    dcost=extractdata(dcost);

%   FDM gradient verification purpose
%
%         dW=0.0001
%         dSE_FDM=[];
%         for i=1:length(Wsdl)
%             Wsdli=Wsdl;
%             Wsdli(i)=Wsdli(i)+dW;
%             [costi, dcost]=dlfeval(@LSE_costfun,Wsdli,Xsbi,Ysbi,LP,NNs,Ws,Bs);
%             dcost_FDM=[dSE_FDM;(costi-cost)/dW];
%         end
%         
%         [dcost dcost_FDM]
end


 function [cost, dcost]=LSE_costfun(Wsdl,Xsn,Ysn,Ystats,Xs,Ys,p,NNs,Ws,Bs,Lamda)
     
     %%%% Reshaping Ws and Bs from Wsdl
     [Ws, Bs]=reshaping_Ws_Bs(Wsdl,Ws,Bs);
     
     %%%% Forward propagation and SE calculation
     [Yp,as,zs]=feedforward(Xs,p,NNs,Ws,Bs,Xsn,Ysn,Ystats);   
%      SE=sum((Yp-Ys).^2);
     % normalized RMSE (NRMSE)
     SE=sqrt(sum((Yp-Ys).^2)/length(Ys));
     NRMSE=SE/(max(Ysn)-min(Ysn));     
     
     
     
     SSw=0;    
     for i=1:length(Ws)
         SSw=SSw+sum(sum(Ws{i}.^2));
     end

     cost=NRMSE+Lamda*SSw;
     %%%% Gradient of SE (dSE) calculation
     dcost = dlgradient(cost,Wsdl);
 end
   

function [Ws, Bs]=reshaping_Ws_Bs(Wsdl,Ws,Bs,extractdata_from_dlarray)
    
    if nargin == 3   % if the number of inputs equals 2
        extractdata_from_dlarray = false; 
    end
    index=1;
    
    for L=1:length(Ws)
        Ws{L}=reshape(Wsdl(index:index+numel(Ws{L})-1),size(Ws{L}));        
        index=index+numel(Ws{L});
        Bs{L}=reshape(Wsdl(index:index+numel(Bs{L})-1),size(Bs{L}));
        index=index+numel(Bs{L});  
        
        if extractdata_from_dlarray
            Ws{L}=extractdata(Ws{L});
            Bs{L}=extractdata(Bs{L});
        end
        
    end
end
  



function y=activation_fun(x)
% 
% %%%% Sigmoid function    
%     for i=1:length(x)
%         y(i)=1/(1+exp(-x(i)));
%     end
%     y=y(:);

 
%%% Tansig function
    y = tansig(x);
    y=y(:);
  
    
% %%%% Rectified linear unit activation Relu function
%     x=dlarray(x);
%     y = relu(x);
%     y=extractdata(y);
%     y=y(:);  
%     
% %%%% Leaky Rectified linear unit activation Relu function
%     x=dlarray(x);
%     y = leakyrelu(x,0.01);
%     y=extractdata(y);
%     y=y(:);    
    
end
     
 function Ysn=test_function_2D_LF01(Xsn)
    %Xsn : size of  Sn by nd

    sizem=size(Xsn);
    Sn=sizem(1);
    nd=sizem(2);

    Ysn=[];
    for i=1:Sn
        xi=Xsn(i,:);
        yi=test_func_LF(xi);
        Ysn=[Ysn;yi];
    end

 end
             
function [Ysn,sYsn,rYsn]=test_function_2D(Xsn)
    %Xsn : size of  Sn by nd

    sizem=size(Xsn);
    Sn=sizem(1);
    nd=sizem(2);

    Ysn=[];
    sYsn=[];
    rYsn=[];
    for i=1:Sn
        xi=Xsn(i,:);
        [yi,sy,ry]=test_func(xi);
        Ysn=[Ysn;yi];
        sYsn=[sYsn;sy];
        rYsn=[rYsn;ry];
    end
end



function [y,sy,ry]=test_func(x)
    x1=x(1);
    x2=x(2);
    y=sin(21*(x1-0.9)^4)*cos(2*(x1-0.9))+(x1-0.7)/2+2*x2^2*sin(x1*x2);
    
    
    sy= 0.00005*(x1+x2);
    randomn=randn(1);
    ry=randomn*sy+y;
    
end

function y=test_func_LF(x)
    x1=x(1);
    x2=x(2);
    y=(test_func(x)-2.0+x1+x2)/(5.0+0.25*x1+0.5*x2); 
end


function [Yp,as,zs]=feedforward(X,LP,NNs,Ws,Bs,Xsn,Ysn,Ystats)
     
     
     Xsize=size(X);
     Sn=Xsize(1); % number of samples
     nd=Xsize(2); % number of dimension
     
     ymin=min(Ysn);
     ymax=max(Ysn);
     
     Yp=zeros(Sn,1);
     
     if isa(Ws{1},'dlarray')
         Yp=dlarray(Yp);
     end
     
     for k=1:Sn
         
         %%%% ===> Input layer
         xk=X(k,:);
         aL=xk(:);
         
%          for u=1:length(xk)
%              xui=ones(size(xk))/2; xui(u)=xk(u);
%              Yui=test_function_2D(xui);
%              aL(u)=Yui;
%          end
         
         
         
         
         zs{1,k}=zeros(size(aL));
         as{1,k}=aL ;   
         
         %%%% ===> Hidden layers forward propagation (L to L+1)
         for L=1:LP-2
            
             wL=Ws{L};
             bL=Bs{L};
             
             zL=wL*aL(:)+bL;
             
             % zL normalization
%              zL=((zL-min(zL))/(max(zL)-min(zL))-0.5)*2;
             aL=activation_fun(zL);
             
             if L==1
                 unifun1=test_function_2D_LF01([xk(1),0.5]);
                 unifun1=(unifun1-Ystats.mean)/Ystats.std;
                 aL(1)=unifun1;
                 
%                  unifun2=test_function_2D([0.7,xk(2)]);
                 unifun2=test_function_2D_LF01([0.5, xk(2)]);
                 unifun2=(unifun2-Ystats.mean)/Ystats.std;
                 aL(2)=unifun2;
   
                 
             end
             if L==2
                 unifun1=test_function_2D_LF01(xk);
                 unifun1=(unifun1-Ystats.mean)/Ystats.std;
                 aL(1)=unifun1;
                 
             end
         
             
             
             zs{L+1,k}=zL;
             as{L+1,k}=aL;
             
         end
         
         %%%% ===> Output layer
         L=LP-1;
         wL=Ws{L};
         bL=Bs{L};
         zL=wL*aL(:)+bL;
         
         %%%% ==> logic or regression
%          aL=activation_fun(zL);
         aL=zL;

         zs{L+1,k}=zL;
         as{L+1,k}=aL;        

         Yp(k)=aL;
         
     end

 end
             
 