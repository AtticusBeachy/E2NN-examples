function [X_stage_1, Y_stage_1] = main_LGMF_free_LF_given_data(xHF, yHF, LFfn,plot_LGMF_details) 


[N_HF, N_dim] = size(xHF);


% h_dists callibrated for data normalized from 0 to 1
h_dists = logspace(log10(0.05*sqrt(N_dim)), log10(0.5*sqrt(N_dim)), 30); % CHANGED TO ACCOMIDATE N DIM

use_convex_hull = true;
LOO_max_pts = 100;%18;%20;
LOO_min_pts = 100;%5;%5;

FNS = {LFfn, @test_function_LINEAR_LF};

xp_bounds = repmat([0                  % rosenbrock
                    1], [1, N_dim]); 



% % GRID SAMPLING
% grid_sampling = true;
% approx_N_pred = 200;%16450; %2000; %200; % approximate number of samples
% res_pred = round(approx_N_pred^(1/N_dim)); % sampling resolution of grid
% xp = gridsamp(xp_bounds, res_pred);
% N_pred = size(xp, 1);

% LHS SAMPLING
grid_sampling = false;
N_pred = 256;%1000;%256;%10000; %sampling resolution
xp = lhsdesign(N_pred, N_dim, 'iterations', 100, 'criterion', 'maximin');
xp = xp_bounds(1,:) + (xp_bounds(2,:) - xp_bounds(1,:)) .* xp;
% RE-SET RANDOMNESS
seed = 1;
rng('default');
rng(seed, 'twister');

% dimension parameters
N_fns = length(FNS);


% get LF functions and uncertainties at HF points
yLF_at_HF = cell(1, N_fns);
sig_LF_at_HF = zeros(N_HF, N_fns); % (only used in LOO)
for i = 1:N_fns
    yLF_at_HFi = zeros(N_HF, 1);
%     sig_LF_at_HFi = zeros(N_HF, 1);
    for j = 1:N_HF
        [val, sig] = FNS{i}(xHF(j,:));
        yLF_at_HFi(j) = val;
        sig_LF_at_HF(j, i) = sig;
    end
    yLF_at_HF{i} = yLF_at_HFi;
end

% get LF values and uncertainties at prediction points
yp_LF = cell(1, N_fns);
sig_xp_LF = zeros(N_pred, N_fns);
for i = 1:N_fns
    yp_LFi = zeros(N_pred, 1);
    for j = 1:N_pred
        [val, sig] = FNS{i}(xp(j,:)) ;% FNS{i}(xp(j)); 
        yp_LFi(j) = val;
        sig_xp_LF(j,i) = sig;
    end
    yp_LF{i} = yp_LFi;
end
% --------------------------- END GET DATA --------------------------------


% ----------------------- CREATE MODEL FOR DATA ---------------------------
LGMF_MODEL.xp = xp;
LGMF_MODEL.N_HF = N_HF;
LGMF_MODEL.N_dim = N_dim;
LGMF_MODEL.N_fns = N_fns;
LGMF_MODEL.h_dists = h_dists;
LGMF_MODEL.use_convex_hull = use_convex_hull;
LGMF_MODEL.LOO_max_pts = LOO_max_pts;
LGMF_MODEL.LOO_min_pts = LOO_min_pts;
LGMF_MODEL.FNS = FNS; % optional

LGMF_MODEL.xHF = xHF;
LGMF_MODEL.yHF = yHF;
LGMF_MODEL.yLF_at_HF = yLF_at_HF;
LGMF_MODEL.sig_LF_at_HF = sig_LF_at_HF;
LGMF_MODEL.sig_xp_LF = sig_xp_LF;
LGMF_MODEL.yp_LF = yp_LF;
% --------------------- END CREATE MODEL FOR DATA -------------------------

% ----------------------------- RUN LGMF ----------------------------------
tic
[LGMF_MODEL] = run_LGMF_free_LF(LGMF_MODEL);
toc
% --------------------------- END RUN LGMF --------------------------------


% --------------------------- FUNCTION OUTPUTS
X_stage_1 = LGMF_MODEL.xp;
Y_stage_1 = LGMF_MODEL.MF1;
% --------------------------- END FUNCTION OUTPUTS


% Basis_plot_styles
% LF_plot_styles
Basis_plot_colors = [
                     1,      0.549,  0     % orange
                     0,      0.5,    0     % darkgreen
                     0.5,    0,      0.5   % purple
                     0.82,   0.706,  0.549 % tan % 1,      0.834,  0     % yellow
                     0.4,    0.4,    0.4   % gray
                     0,      0.5,    0.5   % teal
                     1,      0.834,  0     % yellow %0.82,   0.706,  0.549 % tan
                     0.196,  0.804,  0.196 % lightgreen
                     0.7216, 0.5255, 0.043 % darkgoldenrod
                     0,      1,      1     % cyan
                     ];
LF_plot_colors = Basis_plot_colors;


% ------------------------ BEGIN 1D & 2D PLOTTING -------------------------
% --- 1D

N_pred = length(xp);
N_fns = length(FNS);

% TEST RUN 1D PLOT
% --- 2D
if N_dim == 2 && plot_LGMF_details
    
    % GET HF TO PLOT FOR COMPARISON
    yHF_plot = zeros(size(xp, 1), 1);
    for i = 1:size(xp, 1)
        yHF_plot(i) = HF(xp(i,:));
    end
    
    if grid_sampling
        % RESHAPE RESULTS FOR PLOTTING
        plot_shape = res_pred*ones(1, N_dim);
        x1_plot = reshape(xp(:,1), plot_shape);
        x2_plot = reshape(xp(:,2), plot_shape);
        yp_MF1_plot = reshape(LGMF_MODEL.MF1, plot_shape);
        yHF_grid_plot = reshape(yHF_plot, plot_shape);
        LF_plots = LGMF_MODEL.yp_LF;
        Basis_plots = LGMF_MODEL.BASIS_VALS;
        Basis_plots_2 = LGMF_MODEL.BASIS_VALS_2;
        C_wts_plots = LGMF_MODEL.C_wts;
        
    else
        % CONVERT RESULTS TO GRID FOR PLOTTING
        plot_shape = ceil(N_pred^(1/N_dim)-eps)*ones(1, N_dim);
        x1_plot = linspace(min(xp(:,1)), max(xp(:,1)), plot_shape(1));
        x2_plot = linspace(min(xp(:,2)), max(xp(:,2)), plot_shape(2));
        [x1_plot, x2_plot] = meshgrid(x1_plot, x2_plot);
        yp_MF1_plot = griddata(xp(:,1),xp(:,2),LGMF_MODEL.MF1,x1_plot,x2_plot);
        yHF_grid_plot = griddata(xp(:,1),xp(:,2),yHF_plot,x1_plot,x2_plot);
        % set up Low Fidelity plots
        LF_plots = cell(1, N_fns);
        for i = 1:N_fns
            LF_plot = griddata(xp(:,1),xp(:,2),LGMF_MODEL.yp_LF{i},x1_plot,x2_plot);
            LF_plots{i} = LF_plot(:);
        end
        % set up basis stage 1 plots
        Basis_plots = zeros(prod(plot_shape), N_fns);
        for i = 1:N_fns
            Basis_plot = griddata(xp(:,1),xp(:,2),LGMF_MODEL.BASIS_VALS(:,i),x1_plot,x2_plot);
            Basis_plots(:, i) = Basis_plot(:);
        end

        % set up stage 1 participation functions
        C_wts_plots = zeros(prod(plot_shape), N_fns);
        for i = 1:N_fns
            C_wts_plot = griddata(xp(:,1),xp(:,2),LGMF_MODEL.C_wts(:,i),x1_plot,x2_plot);
            C_wts_plots(:, i) = C_wts_plot(:);
        end
    end
    
    % problem setup
    figure, hold on
    surface(x1_plot, x2_plot, yHF_grid_plot,'FaceColor','r','FaceAlpha',1) % actual red
    scatter3(LGMF_MODEL.xHF(:,1), LGMF_MODEL.xHF(:,2), LGMF_MODEL.yHF,...
             'markerfacecolor','b','markeredgecolor','b')
    LF_legend = cell(1, N_fns);
    for i = 1:N_fns
        LF_plot = reshape(LF_plots{i}, plot_shape);
        mesh(x1_plot, x2_plot, LF_plot, 'FaceColor', LF_plot_colors(i,:),...
            'EdgeColor', LF_plot_colors(i,:),...
            'linestyle','-','linewidth',1.5, 'FaceAlpha',0)%0.3 % LF functions
        LF_legend{i} = ['LF function ',num2str(i)];
    end
    legend(['truth','HF data',LF_legend],'location','northwest','fontsize',12)
    title('LGMF problem setup')
    xlabel('X1')
    ylabel('X2')
    zlabel('Y')
    view([-25, 20]);
    rotate3d
    
    % stage 1 plot
    figure, hold on
    basis_legend = cell(1, N_fns);
    for i = 1:N_fns
        Basis_plot = reshape(Basis_plots(:,i), plot_shape);
        mesh(x1_plot, x2_plot, Basis_plot, 'FaceColor', Basis_plot_colors(i,:),...
            'EdgeColor', Basis_plot_colors(i,:),...
            'linestyle','-','linewidth',1.5, 'FaceAlpha',0) % basis stage 1
        basis_legend{i} = ['Basis function ',num2str(i)];
    end
    surface(x1_plot, x2_plot, yHF_grid_plot,'FaceColor','r','FaceAlpha',1) % actual red
    surface(x1_plot, x2_plot, yp_MF1_plot,'FaceColor','b','FaceAlpha',1) % estimated blue
    scatter3(LGMF_MODEL.xHF(:,1), LGMF_MODEL.xHF(:,2), LGMF_MODEL.yHF,...
             'markerfacecolor','b','markeredgecolor','b') % HF data
    legend([basis_legend,'truth','estimated','HF data'],...
           'location','northwest','fontsize',12)
    title('LGMF prediction')
    xlabel('X1')
    ylabel('X2')
    zlabel('Y')
    view([-25, 20]);
    rotate3d
    
    % --- Participation functions stage #1
    figure, hold on
    basis_legend = cell(1, N_fns);
    for i = 1:N_fns
        C_wts_plot = reshape(C_wts_plots(:,i), plot_shape);
        mesh(x1_plot, x2_plot, C_wts_plot, 'FaceColor', Basis_plot_colors(i,:),...
            'EdgeColor', Basis_plot_colors(i,:),...
            'linestyle','-','linewidth',1.5, 'FaceAlpha',0.3) % basis stage 2
        basis_legend{i} = ['Participation of basis ',num2str(i)];
    end
    legend(basis_legend,'location','northwest','fontsize',12)
    title('LGMF participation functions')
    xlabel('X1')
    ylabel('X2')
    zlabel('Y')
    view([-25, 20]);
    rotate3d
    
end

% ------------------------- END 1D & 2D PLOTTING --------------------------


% ----------------------- PLOT ACTUAL VS PREDICTED ------------------------
% Separate prediction points inside and outside convex hull of HF data
if N_HF > N_dim && N_dim > 1 % possible to make convex hull
    p_in_hull = inhull(LGMF_MODEL.xp, LGMF_MODEL.xHF); % (pts2check, ptsCHull)
else % convex hull impossible (may also use this if C.H. intractable)
    p_in_hull = inbox(LGMF_MODEL.xp, LGMF_MODEL.xHF); % (pts2check, ptsBox)
end

% stage #1 actual vs predicted
if plot_LGMF_details
    % GET HF TO PLOT FOR COMPARISON
    yHF_plot = zeros(size(xp, 1), 1);
    for i = 1:size(xp, 1)
        yHF_plot(i) = HF(xp(i,:));
    end
    figure, hold on
    plot(yHF_plot(~p_in_hull,:), LGMF_MODEL.MF1(~p_in_hull,:),'r+')
    plot(yHF_plot(p_in_hull,:), LGMF_MODEL.MF1(p_in_hull,:),'bo')
    plot(yHF_plot(:), yHF_plot(:), 'k-', 'linewidth', 2)
    title('LGMF actual vs predicted')
    xlabel('Actual')
    ylabel('Predicted')
end

% --------------------- END PLOT ACTUAL VS PREDICTED ----------------------



function [y, sig] = HF(x)
    [y, sig,~]=test_function_2D(x);

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

function [y,sy,ry]=test_func(x)
    x1=x(1);
    x2=x(2);
    y=sin(21*(x1-0.9)^4)*cos(2*(x1-0.9))+(x1-0.7)/2+2*x2^2*sin(x1*x2);
    
    
    sy= 0.00005*(x1+x2);
    randomn=randn(1);
    ry=randomn*sy+y;

function [Ysn, sYsn]=test_function_2D_LF01(Xsn)
    %Xsn : size of  Sn by nd

    sizem=size(Xsn);
    Sn=sizem(1);
    nd=sizem(2);
    
    Ysn=[];
    sYsn = [];
    for i=1:Sn
        xi=Xsn(i,:);
        yi=test_func_LF(xi);
        Ysn=[Ysn;yi];
        sYsn = [sYsn;0];
    end

 function y=test_func_LF(x)
    x1=x(1);
    x2=x(2);
    y=(test_func(x)-2.0+x1+x2)/(5.0+0.25*x1+0.5*x2); 

function [Ysn, sYsn]=test_function_LINEAR_LF(Xsn)
    %Xsn : size of  Sn by nd
    
%     Ysn = sum(Xsn, 2);
    Ysn = zeros(size(Xsn, 1), 1);
    sYsn = zeros(size(Ysn));

