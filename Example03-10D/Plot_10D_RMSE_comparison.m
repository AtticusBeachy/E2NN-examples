% NRMSE
data=[500	0.1346	0.0486
1000	0.1085	0.0284
3000	0.0524	0.0105
5000	0.0297	0.0103]

figure, hold on
plot(data(:,1),data(:,2),'m','LineWidth',2)
plot(data(:,1),data(:,3),'b','LineWidth',2)


scatter(data(:,1), data(:,2),100,'filled',...
            'MarkerEdgeColor','k',...
        'MarkerFaceColor','m')
    
scatter(data(:,1), data(:,3),100,'filled',...
            'MarkerEdgeColor','k',...
        'MarkerFaceColor','b')
    
   
    
    