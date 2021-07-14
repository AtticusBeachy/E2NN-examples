function test_gif
% figure 
% load temp-2d-compared-to-dace_sN40_rstd10p.mat
% % load temp-2d-compared-to-dace_sN100_rstd40p.mat
% S=X;

% scatter3(X(:,1),X(:,2),Y,'filled')

% openfig('Slide22_figure_stochastic_ubds.fig')

openfig('Slide22_figure_deterministic')


set(gcf,'color','w'); % set figure background to white
xlabel('x1')
ylabel('x2')
zlabel('y')
view([120 20])
axis vis3d    
 
for t=120:5:480
    % gif utilities
    view([t,20])
    drawnow;
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    outfile = 'samples_rotation_sN40_deterministic_surfaces.gif';
 
    % On the first loop, create the file. In subsequent loops, append.
    if t==120
        imwrite(imind,cm,outfile,'gif','DelayTime',0,'loopcount',inf);
    else
        imwrite(imind,cm,outfile,'gif','DelayTime',0,'writemode','append');
    end
 
end