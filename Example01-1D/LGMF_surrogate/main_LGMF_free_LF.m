function [Y_stage_1] = main_LGMF_free_LF_given_data(xHF, yHF, LFfn, xp, plot_details) 

% xHF: high-fidelity data locations
% yHF: high-fidelity data responses
% LFfn: low-fidelity function
% xp: points at which a prediction is desired
% plot_details (Boolean): 

[N_HF, N_dim] = size(xHF);


% h_dists callibrated for data normalized from 0 to 1
h_dists = logspace(log10(0.05*sqrt(N_dim)), log10(0.5*sqrt(N_dim)), 30); % CHANGED TO ACCOMIDATE N DIM

use_convex_hull = true;
LOO_max_pts = 20;%100;%20;
LOO_min_pts = 1;%3;%3;%5;%5;

FNS = {LFfn, @test_function_LINEAR_LF};
xp_bounds = repmat([0                  
                    1], [1, N_dim]); 



% % GRID SAMPLING
% grid_sampling = true;
% approx_N_pred = 200;%16450; %2000; %200; % approximate number of samples
% res_pred = round(approx_N_pred^(1/N_dim)); % sampling resolution of grid
% xp = gridsamp(xp_bounds, res_pred);
% N_pred = size(xp, 1);

% LHS SAMPLING
grid_sampling = false;
N_pred = size(xp, 1);
% N_pred = 256;%1000;%256;%10000; %sampling resolution
% xp = lhsdesign(N_pred, N_dim, 'iterations', 100, 'criterion', 'maximin');
xp = xp_bounds(1,:) + (xp_bounds(2,:) - xp_bounds(1,:)) .* xp;
% % RE-SET RANDOMNESS
% seed = 1;
% rng('default');
% rng(seed, 'twister');


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
% Y_stage_2 = LGMF_MODEL.MF2;
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


% -------------------------- BEGIN 1D PLOTTING ----------------------------

N_pred = length(xp);
N_fns = length(FNS);

if N_dim == 1 && plot_details

    % RESHAPE RESULTS FOR PLOTTING
    yp_MF1_plot = reshape(LGMF_MODEL.MF1, size(xp));
    
    % GET HF TO PLOT FOR COMPARISON
    yHF_plot = zeros(N_pred,1);
    for i = 1:N_pred
        yHF_plot(i) = HF(xp(i));
    end
    
    % GET KERNEL TO PLOT
    kerf = @(z) exp(-z.*z/2); % kernel function with max height of 1
    maxHF = max(LGMF_MODEL.yHF); minHF = min(LGMF_MODEL.yHF);
    x0_kerf = mean(xp_bounds, 1);
    distx_kerf = sqrt(sum( (repmat(x0_kerf, N_pred, 1) - xp).^2 , 2 ));
    y_kerf1 = (maxHF-minHF)*kerf(distx_kerf/LGMF_MODEL.h_dist1)+minHF;

    
    % PLOT RESULTS
    
    % problem setup
    figure, hold on
    plot(xp, yHF_plot,'k-','linewidth',2) % actual black
    scatter(LGMF_MODEL.xHF, LGMF_MODEL.yHF,'markerfacecolor','b','markeredgecolor','b')
    plot(xp, y_kerf1,'color',[0.5,0.7,1]) % kernel function
    LF_legend = cell(1, N_fns);
    for i = 1:N_fns
        LF_plot = LGMF_MODEL.yp_LF{i};
        plot(xp, LF_plot, 'color', LF_plot_colors(i,:),'linestyle','-.',...
             'linewidth',1.5) % LF 1
        LF_legend{i} = ['LF function ',num2str(i)];
    end
    legend(['truth','HF data','stage 1 kernel',LF_legend],'location','northwest','fontsize',12)
    title('LGMF problem setup')
    
    % stage 1 plot
    figure, hold on
    basis_legend = cell(1, N_fns);
    for i = 1:N_fns
        Basis_plot = LGMF_MODEL.BASIS_VALS(:,i);
        plot(xp, Basis_plot, 'color', Basis_plot_colors(i,:),'linestyle',':',...
             'linewidth',1.2) % BASIS 1
        basis_legend{i} = ['Basis function ',num2str(i)];
    end
    plot(xp, yHF_plot,'k-','linewidth',2) % actual black
    plot(xp, yp_MF1_plot, 'b-','linewidth',2) % estimated blue
    scatter(LGMF_MODEL.xHF, LGMF_MODEL.yHF,'markerfacecolor','k','markeredgecolor','k')
    legend([basis_legend,'truth','estimated','HF data'],...
           'location','northwest','fontsize',12)
    title('LGMF performance')
    
    % --- Participation functions stage 1
    figure, hold on
    basis_legend = cell(1, N_fns);
    for i = 1:N_fns
        C_wts_plot = LGMF_MODEL.C_wts(:,i);
        plot(xp, C_wts_plot, 'color', Basis_plot_colors(i,:), ...
             'linestyle','-.', 'linewidth',1.5) % BASIS 1
         basis_legend{i} = ['Participation of basis ',num2str(i)];
    end
    legend(basis_legend,'location','northeast','fontsize',12)
    title('Participation functions')
    xlabel('X1')
    ylabel('X2')
    zlabel('Y')
end
% ------------------------- END 1D & 2D PLOTTING --------------------------


% ----------------------- PLOT ACTUAL VS PREDICTED ------------------------
if plot_details
    % Separate prediction points inside and outside convex hull of HF data
    if N_HF > N_dim && N_dim > 1 % possible to make convex hull
        p_in_hull = inhull(LGMF_MODEL.xp, LGMF_MODEL.xHF); % (pts2check, ptsCHull)
    else % convex hull impossible (may also use this if C.H. intractable)
        p_in_hull = inbox(LGMF_MODEL.xp, LGMF_MODEL.xHF); % (pts2check, ptsBox)
    end
    
    % stage 1 actual vs predicted
    figure, hold on
    plot(yHF_plot(~p_in_hull,:), LGMF_MODEL.MF1(~p_in_hull,:),'r+')
    plot(yHF_plot(p_in_hull,:), LGMF_MODEL.MF1(p_in_hull,:),'bo')
    plot(yHF_plot(:), yHF_plot(:), 'k-', 'linewidth', 2)
    title('LGMF Actual vs Predicted')
    xlabel('Actual')
    ylabel('Predicted')
end
% --------------------- END PLOT ACTUAL VS PREDICTED ----------------------


function [y, sig] = HF(x)
    [y, sig]=test_function_1D(x);


function [Ysn, sig]=test_function_1D(Xsn)
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
    sig = zeros(size(Ysn));


function y=test_func(x)
    y=(6*x-2)^2*sin(12*x-4);


function [Ysn, sYsn]=test_function_LINEAR_LF(Xsn)
    %Xsn : size of  Sn by nd
    
%     Ysn = sum(Xsn, 2);
    Ysn = zeros(size(Xsn, 1), 1);
    sYsn = zeros(size(Ysn));



