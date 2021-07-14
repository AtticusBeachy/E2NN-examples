% NRMSE (paper)
data=[100	0.1328	0.0714
200	0.0892	0.0203
300	0.0837	0.0210
500	0.0645	0.0097];

figure, hold on
plot(data(:,1),data(:,2),'m','LineWidth',2)
plot(data(:,1),data(:,3),'b','LineWidth',2)

scatter(data(:,1), data(:,2),100,'filled',...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','m')
    
scatter(data(:,1), data(:,3),100,'filled',...
            'MarkerEdgeColor','k',...
            'MarkerFaceColor','b')




