function E2NN_load_results_Ex_10D_v02_parfor
% Set load_results to "true" to load results for a faster answer. Set it to
% "false" to run the example code (will a few minutes)

% NOTE: accuracy of the E2NN and other fits may vary from run to run
% because no random seed was set.

close all;

%% Training data
Sn=500; %500; %1000; %3000; %5000;

load_results = true; % false
save_results = true; % false

if load_results
    % load temp_result_10D_SN500.mat
    % load temp_result_10D_SN1000.mat
    % load temp_result_10D_SN3000.mat
    % load temp_result_10D_SN5000.mat
    
    load(['temp_result_10D_SN' int2str(Sn) '.mat'])
else
    vali_sample=floor(0.05*Sn)
    % vali_sample=10
    min_cost_target=0.000001
    max_total_epoch=20
    Lamda=0.0000001
    
    
    nd=10;
    Xsn=rand(Sn,nd)*1.1;
    Xsn=lhsdesign(Sn,nd)*1.1;
    
    Sn=length(Xsn(:,1));
    %%%%% ==> Test function
    Ysn=test_function_ND(Xsn);
    
    
    %%%% Ysn normalization
    Ystats.mean=mean(Ysn);
    Ystats.std=std(Ysn);
    Ystats.max=max(Ysn);
    Ystats.min=min(Ysn);
    Ysn=(Ysn-Ystats.mean)/Ystats.std;
    
    
    %% NN model definition
    
    NNs=[nd 50 50 1]; %number of neurons at each layer
    LP=length(NNs);% number of layers
    %
    %  Ws: (Np+1 by Np:  output neuron number by input neuron number
    %  Bs:  Np+1 by 1 :  output neuron number by one
    %
    %% Data Mini-batch setup
    Bn=3000;
    NB=floor(Sn/Bn);
    Bns=[1:NB-1]*Bn;
    Bns=[1 Bns Sn-vali_sample];
    NB=length(Bns)-1;
    
    %% Initializing the Weights
    for p=1:LP-1
        fan_ave=(NNs(p+1)+NNs(p))/2;
        stddev=sqrt(1/fan_ave);
        Ws{p}=randn(NNs(p+1),NNs(p))*stddev+eps;
        Bs{p}=zeros(NNs(p+1),1);
    end
    %% variable flattening to Wsdl
    Wsdl=[];
    for p=1:LP-1
        Wsi=Ws{p};
        Bsi=Bs{p};
        Wsdl=[Wsdl;Wsi(:);Bsi(:)];
    end
    
    
    %% NN mini-batch normalized fitting
    % global opti_history
    % opti_history.x=[];
    % opti_history.fval=[];
    
    
    options = optimoptions('fminunc','SpecifyObjectiveGradient',true,...
        'display','off');
    
    epoch=0;
    repeat_epoch_max=3
    parprocessors=4
    % parpool(parprocessors);
    
    
    minst_cost=inf;
    
    while (epoch<=max_total_epoch && minst_cost>min_cost_target)
        
        parfor proi=1:parprocessors
            %% Initializing the Weights
            for p=1:LP-1
                fan_ave=(NNs(p+1)+NNs(p))/2;
                stddev=sqrt(1/fan_ave);
                Ws{p}=randn(NNs(p+1),NNs(p))*stddev+eps;
                Bs{p}=zeros(NNs(p+1),1);
            end
            %% variable flattening to Wsdl
            Wsdl=[];
            for p=1:LP-1
                Wsi=Ws{p};
                Bsi=Bs{p};
                Wsdl=[Wsdl;Wsi(:);Bsi(:)];
            end
            
            tempid=1;
            
            for repeat_epoch=1:repeat_epoch_max
                
                % Randomizing the data order
                new_order=randperm(Sn);
                Xsn_ppi=Xsn(new_order,:);
                Ysn_ppi=Ysn(new_order);
                
                for bi=1:NB
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Batch 01
                    Ysbi=Ysn_ppi(Bns(bi):Bns(bi+1));
                    Xsbi=Xsn_ppi(Bns(bi):Bns(bi+1),:);
                    
                    
                    Wsdl_opt=fminunc(@net_train_costobj_autograd,Wsdl,options,Xsn_ppi,Ysn_ppi,Ystats,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda);
                    Wsdl=Wsdl_opt;
                    
                    [Ws, Bs]=reshaping_Ws_Bs(Wsdl,Ws,Bs);
                    [Yp,as,zs]=feedforward(Xsn_ppi,LP,NNs,Ws,Bs,Xsn_ppi,Ysn_ppi,Ystats);
                    
                    % normalized RMSE (NRMSE)
                    SE=sum((Yp-Ysn_ppi).^2)/length(Ysn_ppi);
                    NRMSE=sqrt(SE)/(max(Ysn_ppi)-min(Ysn_ppi));
                    
                    SSw=0;
                    for i=1:length(Ws)
                        SSw=SSw+sum(sum(Ws{i}.^2));
                    end
                    cost=NRMSE+Lamda*SSw;
                    
                    temp_str=sprintf('Paralell batch: %d (%d) cost: %f  min_cost: %f'...
                        ,epoch+repeat_epoch, bi, cost, minst_cost);
                    disp(temp_str)
                end
                
            end
            
            cost_pros(proi)=cost;
            Wsdl_pros(:,proi)=Wsdl;
        end % Parallel process ends
        
        [min_SE_pros, proid]=min(cost_pros);
        
        if min_SE_pros<minst_cost
            minst_cost=min_SE_pros;
            best_Wsdl=Wsdl_pros(:,proid);
        end
        epoch=epoch+parprocessors*repeat_epoch_max;
        temp_str=sprintf('Total Epoch: %d (%d) cost: %f  min_cost: %f'...
            ,epoch, 0, min_SE_pros, minst_cost);
        disp(temp_str)
    end
    
    [Ws, Bs]=reshaping_Ws_Bs(best_Wsdl,Ws,Bs);
    
    
    %% Matlab ML function
    net=feedforwardnet(NNs(2:end-1));
    % net.divideParam.trainRatio=0.8;
    % net.divideParam.valRatio=0.2;
    % net.divideParam.testRatio=0;
    
    [net,tr]=train(net, Xsn', Ysn'); %% input_x should be the size of nd-by-Sn
    
    if save_results
        save(['temp_result_10D_SN' int2str(Sn) '.mat'])
    end
end





%% NN result visualization

net=feedforwardnet(NNs(2:end-1));
net.performParam.normalization='standard'
% net.divideParam.trainRatio=0.8;
% net.divideParam.valRatio=0.2;
% net.divideParam.testRatio=0;

[net,tr]=train(net, Xsn', Ysn'); %% input_x should be the size of nd-by-Sn





% testN=1000
% testXsn=lhsdesign(testN,nd);
% Sn=length(testXsn(:,1));

testN=1000
rng(10)
testXsn=rand(testN,nd);
Sn=length(testXsn(:,1));

%%%%% ==> Test function 
testYsn=test_function_ND(testXsn);
[testYp,as,zs]=feedforward(testXsn,LP,NNs,Ws,Bs,Xsn,Ysn,Ystats);
Yp_net=net(testXsn');

testYp=testYp*Ystats.std+Ystats.mean;
Yp_net=Yp_net*Ystats.std+Ystats.mean;


figure, hold on,

plot([min(testYp) max(testYp)],[min(testYp) max(testYp)],'k','LineWidth',1)
scatter(Yp_net,testYsn,5,'filled','MarkerEdgeColor','k','MarkerFaceColor','m')
scatter(testYp,testYsn,5,'filled','MarkerEdgeColor','k','MarkerFaceColor','b')
axis([-2000 15000, 0 15000])


n=length(testYsn)
div=testYsn-testYp;
RMSE_e2nn=sqrt(sum(div.^2)/n)/(max(testYsn)-min(testYsn))
[maxdiv, id]=max(abs(div))



n=length(testYsn)
div=testYsn-Yp_net(:);
RMSE_net=sqrt(sum(div.^2)/n)/(max(testYsn)-min(testYsn))
[maxdiv, id]=max(abs(div))


RMSE_e2nn 
RMSE_net




% 3D surface plots
for i=1:2:nd-1
    for j=i+1:2:nd

        ndi=i;
        ndj=j;

        grid=linspace(0,1,30);
        [xs1,xs2]=meshgrid(grid,grid);
        shape=size(xs1);
        xs1=xs1(:);
        xs2=xs2(:);

        xsnd=ones(length(xs1),nd)*0.5;
        xsnd(:, ndi)=xs1;
        xsnd(:, ndj)=xs2;

        [Yp_bh,as,zs]=feedforward(xsnd,LP,NNs,Ws,Bs,Xsn,Ysn,Ystats);
        Yp_net=net(xsnd');
        Ys=test_function_ND(xsnd);
        Ys_LF01=test_function_ND_LF01(xsnd);

        %%%% ==> De-normalization (back-normalization)
        Yp_bh=(Ystats.std)*Yp_bh+Ystats.mean;
        Yp_net=(Ystats.std)*Yp_net+Ystats.mean;


        xs1=reshape(xs1,shape)*4-2;
        xs2=reshape(xs2,shape)*4-2;
        Ys=reshape(Ys,shape);
        Ys_LF01=reshape(Ys_LF01,shape);
        Yp_bh=reshape(Yp_bh,shape);
        Yp_net=reshape(Yp_net,shape);


        figure, hold on, view([-25 0]), rotate3d
        surface(xs1,xs2,Ys);
        surface(xs1,xs2,Ys_LF01,'FaceColor','g', 'FaceAlpha',0.6)
        surface(xs1,xs2,Yp_bh,'FaceColor','b');
        surface(xs1,xs2,Yp_net,'FaceColor','m')

        % scatter3(Xsn(:,ndi),Xsn(:,ndj),Ysn,80,'filled',...
        %             'MarkerEdgeColor','k',...
        %         'MarkerFaceColor','r')
        view([120,30]) ; 

        title_str=sprintf('%d -D problem: surface %2d -%2d',nd, ndi, ndj);
        title(title_str)
        xlabel('xi')
        ylabel('xj')
    end
end



end
 

%% Imbedded function for opti history
% function stop = opti_iter_output(x,optimValues,state,Xsn,Ysn,Ystats,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda)
% 
% global opti_history
% 
%   stop = false;
%   if isequal(state,'iter')
%     opti_history.x = [opti_history.x  x];
%     opti_history.fval = [opti_history.fval  optimValues.fval];
%   end
% end
     
     
function [cost, dcost]=net_train_costobj_autograd(Wsdl,Xsn,Ysn,Ystats,Ysbi,Xsbi,LP,NNs,Ws,Bs,Lamda)
    
%      Wsdl_opt=fminunc(@net_train_opt_autograd,Wsdl,options,Ysn,Xsn,LP,NNs,Ws,Bs); 
    Wsdl=dlarray(Wsdl);
    [cost, dcost]=dlfeval(@LSE_costfun,Wsdl,Xsn,Ysn,Ystats,Xsbi,Ysbi,LP,NNs,Ws,Bs,Lamda);
    cost=extractdata(cost);
    dcost=extractdata(dcost);
%     min(abs(dcost))

% %   FDM gradient verification purpose
% 
%         dW=0.0001
%         dcost_FDM=[];
%         for i=1:100
%             i
%             Wsdli=Wsdl;
%             Wsdli(i)=Wsdli(i)+dW;
%             [costi, dcost]=dlfeval(@LSE_costfun,Wsdl,Xsn,Ysn,Ystats,Xsbi,Ysbi,LP,NNs,Ws,Bs,Lamda);
%             dcost_FDM=[dcost_FDM;(costi-cost)/dW];
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
     SE=sum((Yp-Ys).^2)/length(Ys);
     NRMSE=sqrt(SE)/(max(Ysn)-min(Ysn));     
     
     
     
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
     
 function Ysn=test_function_ND_LF01(Xsn)
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
             
function [Ysn,sYsn,rYsn]=test_function_ND(Xsn)
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
    
    N=length(x);
    x=x*4-2;
    c1=100;
    c2=1;
    y=0;
    for i=1:N-1
        y=y+c1*(x(i+1)-x(i)^2)^2+c2*(1-x(i))^2;
    end
    
    sy= 0.00005;
    randomn=randn(1);
    ry=randomn*sy+y;
    
end

function y=test_func_LF(x)
   
    N=length(x);
    x=x*4-1.5;
    c1=100;
    c2=1;
    y=0;
    for i=1:N-1
        c1i=c1*(i)/N*1.5;
        c2i=c2+(i)/N*2;
        y=y+c1i*(x(i+1)-x(i)^2)^2+c2i*(1-x(i))^2;
    end

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
                 for i=1:nd
                     xki_uni=ones(1,nd)*0.5;
                     xki_uni(i)=xk(i);
                     unifuni=test_function_ND_LF01(xki_uni);
                     unifuni=(unifuni-Ystats.mean)/Ystats.std;
                     aL(i)=unifuni;
                 end                 
             end
            
              if L==2
                  aL(1)=0;
                 for i=1:nd
                     xki_uni=ones(1,nd)*0.5;
                     xki_uni(i)=xk(i);
                     unifuni=test_function_ND_LF01(xki_uni);
                     unifuni=(unifuni-Ystats.mean)/Ystats.std;
                     aL(1)=aL(1)+unifuni;
                 end   
                 aL(1)=aL(1)/nd;
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
             
 