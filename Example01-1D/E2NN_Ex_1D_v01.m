function E2NN_Ex_1D_v01
% NOTE: accuracy of the standard NN and E2NN fits will vary from run to run
% because no random seed was set.
% To see additional details of the LGMF model building process, change 
% "plot_LGMF_details" from "false" to "true"
%% paths
addpath('cokriging-master')
addpath('cokriging-master/dace')
addpath('LGMF_surrogate')
%% Training data
close all
Sn=3
vali_sample=floor(0.05*Sn)
vali_sample=0
min_SE_target=0.000001
max_epoch=30
restart_epoch=5

nd=1;
% Xsn=lhsdesign(Sn,nd);
Xsn=[0 0.5 1.0]';
Sn=length(Xsn(:,1));
%%%%% ==> HF test function 
Ysn=test_function_1D(Xsn);

Xsn_original = Xsn;
Ysn_original = Ysn;

%%%% Ysn normalization 
Ymean=mean(Ysn);
Ystd=std(Ysn);
Ysn=(Ysn-Ymean)/Ystd;




%% NN model definition

Lamda=0.000;
NNs=[nd 10 10 1]; %number of neurons at each layer
LP=length(NNs);% number of layers
%
%  Ws: (Np+1 by Np:  output neuron number by input neuron number
%  Bs:  Np+1 by 1 :  output neuron number by one
%
%% Data Mini-batch setup
Bn=22;
NB=floor(Sn/Bn);
Bns=(1:NB-1)*Bn;
Bns=[1 Bns Sn-vali_sample];
NB=length(Bns)-1;


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


%% E2NN mini-batch normalized fitting
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
        
        
        Wsdl_opt=fminunc(@net_train_costobj_autograd,Wsdl,options,Xsn,Ysn,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda,Ymean,Ystd);
        Wsdl=Wsdl_opt;

        [Ws, Bs]=reshaping_Ws_Bs(Wsdl,Ws,Bs);
        [Yp,as,zs]=feedforward(Xsn,LP,NNs,Ws,Bs,Xsn,Ysn,Ymean,Ystd);  

         % normalized RMSE (NRMSE)
         SE=sum((Yp-Ysn).^2)/length(Ysn);
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


%% Matlab NN function
net=feedforwardnet([2,2]);%NNs(2:end-1));
net.divideParam.trainRatio=1.0;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0;

[net,tr]=train(net, Xsn', Ysn'); %% input_x should be the size of nd-by-Sn

%% Kriging and CoKriging

krig = dacefit(Xsn,Ysn,@regpoly0,@corrgauss,1e-1,1e-6,3);
Xsn_lf = linspace(0, 1, 200);
Xsn_lf = Xsn_lf(:);
Ysn_lf = test_function_1D_LF01(Xsn_lf);
Ysn_lf=(Ysn_lf-Ymean)/Ystd;
[dmodel, dmc, dmd] = cokriging2(Xsn_lf,Ysn_lf, Xsn, Ysn,@regpoly0,@corrgauss,1e-3,30);


%% Result visualization and comparison

% iteration history
figure, semilogy(opti_history.fval)

xs1=linspace(0,1.0,60);
xs1=xs1(:);

% HF model:
Ys=test_function_1D(xs1);
% LF model:
Ys_LF01=test_function_1D_LF01(xs1);

% Model predictions:
% E2NN
[Yp_bh,as,zs]=feedforward(xs1,LP,NNs,Ws,Bs,Xsn,Ysn,Ymean,Ystd);
% NN
Yp_net=net(xs1');

% Kriging
kg = predictor(xs1,krig);
% Cokriging
cok = predict_cok2(xs1, dmodel);

% LGMF 
Ysn_unscaled = (Ystd)*Ysn+Ymean;
plot_LGMF_details = false;%true;
[lgmf] = main_LGMF_free_LF(Xsn_original,Ysn_original, @test_function_1D_LF01, xs1, plot_LGMF_details);

%%%% ==> De-normalization (back-normalization)
Yp_bh=(Ystd)*Yp_bh+Ymean;
Yp_net=(Ystd)*Yp_net+Ymean;
Ysn=(Ystd)*Ysn+Ymean;
kg=(Ystd)*kg+Ymean;
cok=(Ystd)*cok+Ymean;

 
% Plot HF and LF models
figure, hold on,
plot(xs1,Ys, 'k-', 'linewidth',2);
plot(xs1,Ys_LF01, '-.', 'color',[0 0.5 0], 'linewidth',2)
scatter(Xsn,Ysn,100,'filled',...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','r')
legend('HF','LF','HF samples','Location','NorthWest','FontSize', 12)
legend boxoff

% Compare Kriging and CoKriging
figure, hold on,
plot(xs1, Ys,'k', 'linewidth',2)
scatter(Xsn,Ysn,100,'filled','MarkerEdgeColor','k','MarkerFaceColor','r')     
plot(xs1, kg,'m--', 'linewidth',2)
plot(xs1, cok,'color',[1.00,0.41,0.16], 'linewidth',2)
plot(xs1, lgmf, 'g--', 'linewidth',2)
legend('HF function','HF samples','Kriging','Co-Kriging','LGMF',...
       'Location','NorthWest', 'FontSize', 12)
legend boxoff


% Compare NN and E2NN
figure, hold on,
plot(xs1,Ys, 'k', 'linewidth',2);
plot(xs1,Yp_net,'color','m', 'linewidth',2);
plot(xs1,Yp_bh,'color','b', 'linewidth',2);
scatter(Xsn,Ysn,100,'filled',...
            'MarkerEdgeColor','k',...
        'MarkerFaceColor','r')
legend('HF','std. NN','E2NN','HF samples','Location','NorthWest','FontSize', 12)
legend boxoff


opti_history;


end
 

%% Imbedded function for opti history
function stop = opti_iter_output(x,optimValues,state,Xsn,Ysn,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda,Ymean,Ystd)

global opti_history

  stop = false;
  if isequal(state,'iter')
    opti_history.x = [opti_history.x  x];
    opti_history.fval = [opti_history.fval  optimValues.fval];
  end
end
     
     
function [cost, dcost]=net_train_costobj_autograd(Wsdl,Xsn,Ysn,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda,Ymean,Ystd)
    
% % %      Wsdl_opt=fminunc(@net_train_opt_autograd,Wsdl,options,Ysn,Xsn,LP,NNs,Ws,Bs); 
    Wsdl=dlarray(Wsdl);
    [cost, dcost]=dlfeval(@LSE_costfun,Wsdl,Xsn,Ysn,Xsbi,Ysbi,LP,NNs,Ws,Bs,Lamda,Ymean,Ystd);
    cost=extractdata(cost);
    dcost=extractdata(dcost);
% 
% % %   FDM gradient verification purpose
% 
%         dW=0.00002;
%         dcost_FDM=[];
%         for i=1:length(Wsdl)
%             Wsdli=Wsdl;
%             Wsdli(i)=Wsdli(i)+dW;
%             [costi, dcost]=dlfeval(@LSE_costfun,Wsdli,Xsn,Ysn,Xsbi,Ysbi,LP,NNs,Ws,Bs,Lamda,Ymean,Ystd);
%             costi=extractdata(costi);
%             dcost_FDM=[dcost_FDM;(costi-cost)/dW];
%         end
%         dcost=dcost_FDM;
%         
% %         
% %         [dcost dcost_FDM]
end


 function [cost, dcost]=LSE_costfun(Wsdl,Xsn,Ysn,Xs,Ys,p,NNs,Ws,Bs,Lamda,Ymean,Ystd)
     
     %%%% Reshaping Ws and Bs from Wsdl
     [Ws, Bs]=reshaping_Ws_Bs(Wsdl,Ws,Bs);
     
     %%%% Forward propagation and SE calculation
     [Yp,as,zs]=feedforward(Xs,p,NNs,Ws,Bs,Xsn,Ysn,Ymean,Ystd);   
%      SE=sum((Yp-Ys).^2);
     % normalized RMSE (NRMSE)
     SE=sum((Yp-Ys).^2)/length(Ys);
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
     
 function [Ysn, Ssn]=test_function_1D_LF01(Xsn)
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
    Ssn = zeros(size(Ysn));
 end
             
function Ysn=test_function_1D(Xsn)
    %Xsn : size of  Sn by nd

    sizem=size(Xsn);
    Sn=sizem(1);
    nd=sizem(2);

    Ysn=[];
    for i=1:Sn
        xi=Xsn(i,:);
        yi=test_func(xi);
        Ysn=[Ysn;yi];
    end
end



function y=test_func(x)
%     x1=x(1);
%     x2=x(2);
% %     y=x1^1.5-3*x1^2*x2+0.5*sin(x2)^2-sin(x1*x2)-2*x1*x2^2;   
% %     y=2*x1-3*x1^2+3*sin(3*x2)^2-x2^2.6+10*x1*x2;  
% %     y=2*x1-3*x1^2+3*sin(3*x2)^2-x2^2.6; 
% %     y=2*x1-3*x1^2+3*sin(3*x2)^2-x2^2.6; 
% %     y=(2*x1-3*x1^2)+(2*x2-3*x2^2)+0.1*x1*x2-0.12*x1^2/(x2+0.2);
%     y=(2*x1-3*x1^2)+(2*x2-3*x2^3);
    y=(6*x-2)^2*sin(12*x-4);


end

function y=test_func_LF(x)
    y=0.5*test_func(x)+10*(x-0.5)-5; 
end


function [Yp,as,zs]=feedforward(X,LP,NNs,Ws,Bs,Xsn,Ysn,Ymean,Ystd)
     
     
     Xsize=size(X);
     Sn=Xsize(1); % number of samples
     nd=Xsize(2); % number of dimension
     
     ymin=min(Ysn);
     ymax=max(Ysn);
     
     
     ylf=test_function_1D_LF01([0:0.05:1.0]');
     Ymean=mean(ylf);
     Ystd=std(ylf);
%      Ysn=(Ysn-Ymean)/Ystd;
     
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
%                  unifun1=test_function_1D_LF01(xk);
%                  aL(1)=unifun1;
%                  unifun=test_function_2D(xk)+2*xk(1)-xk(2);
%                  aL(1)=unifun;    
                 
             end
             if L==2
                 unifun1=test_function_1D_LF01(xk);
%                  unifun1=(unifun1-Ymean)/Ystd;
                 aL(1)=unifun1;
% %                  unifun1=as{L,k}(1);
%                  unifun2=as{L,k}(2);
%                  aL(1)=unifun1+unifun2;                
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
             
 