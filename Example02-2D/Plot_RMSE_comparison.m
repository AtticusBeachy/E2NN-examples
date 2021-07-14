data1=[24  0.0864];

data2=[24	0.065
30	0.058
40	0.016
60	0.0051];

data3=[24	0.026
30	0.017
40	0.0037
60	0.0025];

data4=[12	0.0251	
24	0.0064	
30	0.0047	
40	0.0024	
60	0.0013	
100	0.0009];

data5=[12	0.0076
24	0.0065
30	0.0050
40	0.0055
60	0.0066
100	0.0046];





figure, hold on
set(gca, 'YScale', 'log')
set(gca,'FontSize',16)
plot(data2(:,1), data2(:,2),'g','LineWidth',2,'handlevisibility','off')
plot(data3(:,1), data3(:,2),'c','LineWidth',2,'handlevisibility','off')
plot(data4(:,1), data4(:,2),'b','LineWidth',2,'handlevisibility','off')
plot(data5(:,1), data5(:,2),'r','LineWidth',2,'handlevisibility','off')
%
scatter(data1(:,1), data1(:,2),100,'filled',...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','m') 
%
scatter(data2(:,1), data2(:,2),100,'filled',...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','g')
%
scatter(data3(:,1), data3(:,2),100,'filled',...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','c') 
%
scatter(data4(:,1), data4(:,2),100,'filled',...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','b')
   
%
scatter(data5(:,1), data5(:,2),100,'filled',...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor','r') 

legend('LOC Kriging', 'ALOS (HF only)','ALOS (MF)','E2NN','E2NN with LGMF',...
       'Location','NorthEast')

xlim([10, 60])
ylim([6e-4, 3e-1])
legend boxoff