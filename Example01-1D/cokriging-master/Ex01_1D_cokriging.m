function Ex01_1D_cokriging

    clc
close all
    addpath('dace')
    
    Xsn=[0 0.5 1.0]';
    Xsn=[0:0.5:1.0]';
    Sn=length(Xsn(:,1));
    %%%%% ==> Test function 
    Ysn=test_function_1D(Xsn);
    
    Xsn_lf=linspace(0, 1, 200)';
    Ysn_lf=test_function_1D_LF01(Xsn_lf);

  

    [dmodel, dmc, dmd] = cokriging2(Xsn_lf,Ysn_lf, Xsn, Ysn,@regpoly0,@corrgauss,1e-3,30);
    krig = dacefit(Xsn,Ysn,@regpoly0,@corrgauss,1e-1,1e-6,3);

    xs = gridsamp([0;1],50);
    xs=xs(:);
    ys = test_function_1D(xs);
    cok = predict_cok2(xs, dmodel);
    kg = predictor(xs,krig);

    figure, hold on,
%     clf;
    % Co-Kriging
 
    
    true_plot = sortrows([xs, ys],1);
    cok_plot = sortrows([xs, cok],1);
    kg_plot = sortrows([xs, kg],1);
    plot(true_plot(:,1), true_plot(:,2),'k')
    scatter(Xsn,Ysn,100,'filled','MarkerEdgeColor','k','MarkerFaceColor','r')     
    plot(cok_plot(:,1), cok_plot(:,2),'b')
    plot(kg_plot(:,1), kg_plot(:,2),'m--')

    legend('HF function','Samples','Co-Kriging','Kriging','Location','SouthWest')
    legend boxoff
% 
%     y1 = hf(xs(:,1));
%     y2 = cok;
%     err(3) = sqrt(mean((y2 - y1).^2))/mean(y1);
% 
%     axis off
%     grid off
% 
%     set(gcf,'papersize',[4 3],'paperposition',[0 0 4 3])
%     saveas(gcf,'krigingExample.pdf')

end

 function Ysn=test_function_1D_LF01(Xsn)
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
