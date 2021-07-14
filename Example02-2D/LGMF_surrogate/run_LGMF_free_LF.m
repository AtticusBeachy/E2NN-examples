function [LGMF_MODEL] = run_LGMF_free_LF(LGMF_MODEL)

% -------------------------- EXTRACT DATA VALUES --------------------------
xp = LGMF_MODEL.xp;
N_HF = LGMF_MODEL.N_HF;
N_dim = LGMF_MODEL.N_dim;
N_fns = LGMF_MODEL.N_fns;
h_dists = LGMF_MODEL.h_dists;
use_convex_hull = LGMF_MODEL.use_convex_hull;
LOO_max_pts = LGMF_MODEL.LOO_max_pts;
LOO_min_pts = LGMF_MODEL.LOO_min_pts;

xHF = LGMF_MODEL.xHF;
yHF = LGMF_MODEL.yHF;
yLF_at_HF = LGMF_MODEL.yLF_at_HF;
sig_LF_at_HF = LGMF_MODEL.sig_LF_at_HF;
sig_xp_LF = LGMF_MODEL.sig_xp_LF;
yp_LF = LGMF_MODEL.yp_LF;

N_pred = size(xp, 1);
% ------------------------ END EXTRACT DATA VALUES ------------------------


% ------------------ CHOOSE WHICH POINTS TO LEAVE OUT ---------------------
if N_dim == 1
    % only leave out ends
    Kvp = [1, N_HF]';
    loo_i = (2:N_HF-1)'; if isempty(loo_i); loo_i = N_HF; end
elseif use_convex_hull
    if N_dim < 4
        K = convhull(xHF);
    else
        K = convhulln(xHF);
    end
    Kvp = unique(K(:)); % convex hull vertex point indices
    loo_i = setdiff(1:N_HF, Kvp)'; % Indices of all pts inside convex hull
else % multiple dimensions but don't use convex hull
    % pick points randomly without using a convex hull
    N_loo = min(N_HF, LOO_max_pts);
    loo_i = datasample((1:N_HF)', N_loo, 'Replace', false);
end

% CORRECT NUMBER OF POINTS INSIDE CONVEX HULL IF NECESSARY
if length(loo_i) < LOO_min_pts % too few points inside convex hull
    N_loo = min(LOO_min_pts, N_HF);
    N_add = N_loo - length(loo_i);
    N_add = min(N_add, length(Kvp)); % crash if try add > vertex pts than exist
    loo_i_add = datasample(Kvp,N_add,'Replace',false);
    loo_i = [loo_i; loo_i_add];
elseif length(loo_i) > LOO_max_pts % too many points inside convex hull
    loo_i = datasample(loo_i,LOO_max_pts,'Replace',false);
end
N_loo = length(loo_i)
% ---------------- END CHOOSE WHICH POINTS TO LEAVE OUT -------------------


% -------------- STAGE ONE CORRELATION LENGTH OPTIMIZATION ----------------

if N_fns == 1
    % only 1 LF function: skip stage #1
    % MF1 is LF1 (the only LF function)
    MF1 = yp_LF{1};
    sig_MF1 = zeros(N_pred, 1);
    C_wts = ones(N_pred, N_fns); % only for plot
    C_wts_norm = ones(N_pred, N_fns); % only for plot
    BASIS_VALS = yp_LF{1}; % only for plot
    
    % additional data for stage #2
    MF1_at_HF = yLF_at_HF{1};
    sig_MF1_at_HF = zeros(N_HF, N_fns); % if LF has NDK fit w/ epsitemic uncertainty, include vals?
    sig_MF1_at_p = zeros(N_pred,1); % aleatory, therefore 0
    
else % run stage #1
    % Run LOO to get data on correlation distances
    [h_dists_stable, h_dist_avg_loo_RMSEs, ...
        feasible_domain_outliers, ~] = ...
        LOO_for_h_dist_opt(@LGMF_stage1, h_dists, loo_i, xHF, yHF, ...
        yLF_at_HF, sig_LF_at_HF, true);
    
    % get feasible domain
    feasible_domain = find(feasible_domain_outliers & h_dists_stable);
    
    
    if isempty(feasible_domain)
        % If COND constraint is satsified anywhere, OUTLIERS constraint will be
        % satisfied somewhere in that domain
        % Therefore, lack of a feasible domain means COND constraint failed
        % everywhere
        disp('WARNING: all correlation distances in stage #1 fail cond check during LOO')
        
        % Re-run LOO to get additional data on correlation distances
        [~, h_dist_avg_loo_RMSEs, ...
            feasible_domain_outliers, num_unstable_vals] = ...
            LOO_for_h_dist_opt(@LGMF_stage1, h_dists, loo_i, xHF, yHF, ...
            yLF_at_HF, sig_LF_at_HF, false);
        % get feasible domain again
        num_unstable_vals(~feasible_domain_outliers) = Inf;
        feasible_domain = num_unstable_vals == min(num_unstable_vals);
    end
    
    
    % get optimal correlation distance
    h_dists_1 = h_dists(feasible_domain);
    h_dist_avg_loo_RMSEs_1 = h_dist_avg_loo_RMSEs(feasible_domain);
    h_dist1_idxs = find(h_dist_avg_loo_RMSEs_1 == min(h_dist_avg_loo_RMSEs_1)); % get opt indices
    h_dist1_idx = h_dist1_idxs(ceil(length(h_dist1_idxs)/2)); % get middle opt index
    h_dist1 = h_dists_1(h_dist1_idx)
    
    % ------------- END STAGE ONE CORRELATION LENGTH OPTIMIZATION -------------
    
    % --------------------------- RUN STAGE #1 --------------------------------
    [MF1, sig_MF1, C_wts, C_wts_norm, BASIS_VALS, ~, WEIGHT_ERRs, SINGULAR_ERRs, ~] = ...
        LGMF_stage1(xp, xHF, yHF, yLF_at_HF, yp_LF, sig_xp_LF, h_dist1, false);
    % ------------------------- END RUN STAGE #1 ------------------------------
    
    % display warnings
    if any(WEIGHT_ERRs)
        disp('WARNING: Stage #1 fit suffers from weight error')
    end
    if any(SINGULAR_ERRs)
        disp('WARNING: Stage #1 fit suffers from singularity error')
    end
    
    % ADD QUANTITIES TO MODEL
    % correlation related
    %LGMF_MODEL.h_dists = h_dists;
    LGMF_MODEL.feasible_domain_outliers = feasible_domain_outliers;
    LGMF_MODEL.h_dists_stable = h_dists_stable;
    LGMF_MODEL.feasible_domain = feasible_domain;
    LGMF_MODEL.h_dists_1 = h_dists_1;
    LGMF_MODEL.h_dist_avg_loo_RMSEs_1 = h_dist_avg_loo_RMSEs_1;
    LGMF_MODEL.h_dist1 = h_dist1;
    
end

% record stage #1 outputs
LGMF_MODEL.MF1 = MF1;
LGMF_MODEL.sig_MF1 = sig_MF1;
LGMF_MODEL.C_wts = C_wts;
LGMF_MODEL.C_wts_norm = C_wts_norm;
LGMF_MODEL.BASIS_VALS = BASIS_VALS;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ----------------------------- FUNCTIONS ---------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [h_dists_stable, h_dist_avg_loo_RMSEs, ...
    feasible_domain_outliers, num_unstable_vals] = ...
    LOO_for_h_dist_opt(LGMF_stage_1_or_2, h_dists, loo_i, xHF, yHF, ...
                          yLF_at_HF, sig_LF_at_HF, FAST)

N_fns = length(yLF_at_HF); %number of functions
N_HF = length(yHF); %number of HF data points
N_loo = length(loo_i); %number of points to leave out

% initialize values
h_dist_avg_loo_RMSEs = zeros(length(h_dists), 1);
I_outliers = zeros(length(h_dists),1);
if FAST
    h_dists_stable = true(length(h_dists), 1);
    num_unstable_vals = [];
else
    h_dists_stable = []; % I never run slow optimization unless all unstable
    num_unstable_vals = 0;
end

% FAST LOO CHECK (TERMINATE IF INSTABILITY OCCURS)
for i = 1:length(h_dists) % go thru correlation distances
    h_dist = h_dists(i);
    net_loo_rmse = 0; 
    Isum = 0;
    worst_cond_val = 0; % only used when FAST is false
    for j = 1:N_loo % go thru pts. to leave out
        % leave one out
        loo_idx = loo_i(j); %index to leave out
        xp_loo = xHF;
        xHF_loo = xHF; xHF_loo(loo_idx, :) = [];
        yHF_loo = yHF; yHF_loo(loo_idx) = [];
        yLF_at_HF_loo = yLF_at_HF; for k = 1:N_fns; yLF_at_HF_loo{k}(j) = []; end
        yp_LF_loo = yLF_at_HF;
        sig_xp_LF_loo = sig_LF_at_HF;
        
        % get stage #1 fit
        [MF_loo, sig_MF_loo, ~, ~, ~, TERMINATED, ~, ~, num_unstable] = ...
               LGMF_stage_1_or_2(xp_loo, xHF_loo, yHF_loo, yLF_at_HF_loo,...
               yp_LF_loo, sig_xp_LF_loo, h_dist, FAST);
        
        if FAST && TERMINATED % termination requires fast be true
            h_dists_stable(i) = false;
            net_loo_rmse = Inf; % maximum error
            Isum = N_HF * N_loo; % all points left out
            break
        end
        
%         if ~FAST && worst_cond_case > worst_cond_val
%             worst_cond_val = worst_cond_case;
%         end
        % get and sum root mean squared error of fit
        loo_rmse = sqrt( mean( (MF_loo - yHF).^2 ) );
        net_loo_rmse = net_loo_rmse + loo_rmse;
        % Record HF samples outside of 3*sigma error bounds
        Isum = Isum + sum( (MF_loo-yHF).^2 - 9*sig_MF_loo.^2 > 0);
    end
    % add errors to list
    h_dist_avg_loo_RMSEs(i) = net_loo_rmse/N_loo;
    
    % add number of outliers to list
    I_outliers(i) = Isum;
    
    % add worst condition values to list
    if ~FAST
        num_unstable_vals = num_unstable_vals + num_unstable;
    end
end
% get feasible domain due to outlier constraint
% (no more than 1% of evaluated points may be outliers, rounded up)
feasible_domain_outliers = I_outliers <= ceil(0.01*N_loo*(N_loo-1));
% fix feasible domain if empty
if sum(feasible_domain_outliers) == 0 % no feasible points
    % only h_dist's w/ minimum # of outliers are feasible
    feasible_domain_outliers = I_outliers == min(I_outliers);
end



function [yp_MF1, sig_MF1, C_wts, C_wts_norm, BASIS_VALS, TERMINATED, WEIGHT_ERRs, SINGULAR_ERRs, num_unstable] = ...
               LGMF_stage1(xp, xHF, yHF, yLF_at_HF, yp_LF, sig_xp_LF, h_dist, TERMINATE_IF_ERR, WEIGHT_TOL, EQUAL_TOL, SINGULAR_TOL)

% --------------------------- VARIABLE MEANINGS ---------------------------
% --- INPUTS ---
% xp - prediction points
% xHF - High Fidelity data locations
% yHF - High Fidelity data responses
% yLF_at_HF - Low Fidelity data at High Fidelity locations
% yp_LF - Low Fidelity values at prediction points (may come from NDK)
% sig_p_LF - standard deviation of epistemic uncertainty of LF values at
%            prediction points
% h_dist - the correlation distance of the kernel function values
% 
% --- OUTPUTS ---
% yp_MF1 - LGMF stage 1 predictions
% sig_MF1 - standard deviation of the uncertainty of LGMF stage 1
%           predictions
% Cwt - participation function weights found by Localized Galerkin method
% 
% --- OTHER ---
% wt_dy (k_wt) - kernel weights of HF data locations
% ------------------------- END VARIABLE MEANINGS -------------------------

% initialize tolerances if unset
if ~exist('WEIGHT_TOL','var') || isempty(WEIGHT_TOL)
    WEIGHT_TOL = eps;
end

if ~exist('EQUAL_TOL','var') || isempty(EQUAL_TOL)
    EQUAL_TOL = 1*eps;
end

if ~exist('SINGULAR_TOL','var') || isempty(SINGULAR_TOL)
    SINGULAR_TOL = 1/eps;%10^5;
end

% solve for dimension parameters
%[N_HF, N_dim] = size(xHF); 
N_fns = length(yLF_at_HF);
N_pred = length(xp); 

% solve for deviations
dy_LF_at_HF = cell(1, N_fns);
for i = 1:N_fns
    dy_LF_at_HF{i} = yHF - yLF_at_HF{i}; % additive corrections
end

% INITIALIZE OUTPUTS
yp_MF1 = zeros(N_pred, 1);
sig_MF1 = zeros(N_pred, 1);
C_wts = zeros(N_pred, N_fns);
C_wts_norm = zeros(N_pred, N_fns);
% dMF1_dBasis = zeros(N_pred, N_fns);
TERMINATED = false;
BASIS_VALS = zeros(N_pred, N_fns);
WEIGHT_ERRs = false(N_pred, 1);
SINGULAR_ERRs = false(N_pred, 1);
num_unstable = 0;

% CYCLE THRU ALL PREDICTION POINTS
for i = 1:N_pred
    xi = xp(i,:);
    
    % GET HF LOCATION WEIGHTS 
    wt_dy = get_HF_data_weights(xi, xHF, h_dist);
    
    % CHECK FOR WEIGHT ERRORS
    wt_within_TOL = wt_dy > WEIGHT_TOL;
    WEIGHT_ERR = sum(wt_within_TOL) == 1; % err if only one weight captured
    WEIGHT_ERRs(i) = WEIGHT_ERR;
    
    % TERMINATION CHECK
    if WEIGHT_ERR && TERMINATE_IF_ERR
        TERMINATED = true;
        %disp('Stage #1 terminated due to failing weight check')
        break
    end
    
    % initialize basis values and uncertainty
    basis_at_HF = cell(1, N_fns);
    basis_at_xp = zeros(N_fns,1);
    sig_xp_basis_discrep = zeros(N_fns,1);
    dMF1_dBasis_i = zeros(1, N_fns);
    
    % get design matrix at HF points
    X_fit = get_design_matrix(xHF, 'interactions');
    
    % get design matrix at prediction point
    Xp_des = get_design_matrix(xi, 'interactions');
    
    % SOLVE FOR BASIS FUNCTION VALUES
    for j = 1:N_fns
        % ADDITIVE CORRECTIONS
        % --- SOLVE FOR LEAST SQUARES FIT OF LF ERRORS AT HF POINTS
        [B_fit,~,MSE] = lscov(X_fit, dy_LF_at_HF{j}, wt_dy + eps);
        % --- SOLVE FOR BASIS AT HF POINTS
        basis_at_HF{j} = yLF_at_HF{j} + X_fit*B_fit;
        % --- SOLVE FOR BASIS AT PREDICTION POINT
        basis_at_xp(j) = yp_LF{j}(i) + Xp_des*B_fit;
        sig_xp_basis_discrep(j) = sqrt(MSE);
    end
    
    
    % CHECK FOR EQUALITY ERRORS (DUPLICATE BASIS FUNCTIONS FOR POINT)
    duplicate_basis_idx = []; % indices of duplicate basis functions to eliminate
    
    for m = 1:N_fns
        mth_weighted_hf_vals = basis_at_HF{m}(wt_within_TOL);
        for n = m+1:N_fns
            nth_weighted_hf_vals = basis_at_HF{n}(wt_within_TOL);
            %differences = abs(mth_weighted_hf_vals - nth_weighted_hf_vals);
            ratios = abs(mth_weighted_hf_vals./nth_weighted_hf_vals);
            % compare basis functions to check if all
            %basis_identical = all(differences < EQUAL_TOL);
            basis_identical = all(abs(ratios-1) < EQUAL_TOL);
            if basis_identical
                duplicate_basis_idx = [duplicate_basis_idx, n];
                % fprintf('Equality error: basis fns %d and %d equal', m, n)
            end
        end
    end
    
    % ELIMINATE DUPLCIATE BASIS FUNCTIONS (IF ANY)
    basis_at_HF(:,duplicate_basis_idx) = []; 
    
    % SOLVE LOCALIZED GALERKIN FOR PARTICIPATION FUNCTIONS (C VALUES) AND CHECK FOR ERROR
    [C_wt, SINGULAR_ERR, ~] = solve_localized_galerkin_at_point(yHF, basis_at_HF, wt_dy, SINGULAR_TOL);
    % record singularity details
    SINGULAR_ERRs(i) = SINGULAR_ERR;
    if SINGULAR_ERR
        num_unstable = num_unstable + 1;
    end
    
    
    % INSERT ZERO WEIGHTS OF DUPLCIATE BASIS FUNCTIONS (IF ANY)
    duplicate_basis_idx = unique(duplicate_basis_idx);
    for j = 1:length(duplicate_basis_idx)
        idx = duplicate_basis_idx(j);
        C_wt = [C_wt(1:idx-1); 0; C_wt(idx:end)];
    end
    
    % TERMINATION CHECK
    if SINGULAR_ERR && TERMINATE_IF_ERR
        TERMINATED = true;
        %disp('Stage #1 terminated due to failing singularity check (despite passing weight check)')
        break
    end
    
%     % GET NORMALIZED WEIGHTS
%     scale_about = 1/length(C_wt);
%     d = C_wt - scale_about;
%     d = d*scale_about/max(-[d;-1/length(C_wt)]); %only scale if min(C_wt<0)
%     C_wt_norm = d + scale_about; 
    % ALTERNATIVE: DON'T NORMALIZE - ajb May 28, 2021
    C_wt_norm = C_wt;
    
    % modify uncertainties of LF fns (sig_xp_LF) to uncertainties of basis
    % functions (sig_xp_basis)
    % additive uncertainties (no change)
    sig_xp_basis_aleatory = sig_xp_LF;
    
    % GET PREDICTION UNCERTANTY
    % - from discrepancies in the errors at HF points
    %   assume errors are uncorrelated (will not be the case if the HF
    %   function has significant nonlinearity that is not captured by any
    %   of the LF functions)
    sig_discrepancy = norm(C_wt.*sig_xp_basis_discrep); 
    % - from aleatory uncertainty in the LF/basis functions
    %   assume errors are uncorrelated (will not be the case if NDK uses
    %   the same sample locations for all functions)
    sig_aleatory = norm(C_wt.*sig_xp_basis_aleatory(i,:)');
    
    % RECORD OUTPUTS
    yp_MF1(i) = sum(C_wt.*basis_at_xp);
    sig_MF1(i) = hypot(sig_discrepancy, sig_aleatory);
    C_wts(i,:) = C_wt';
    C_wts_norm(i,:) = C_wt_norm';
    BASIS_VALS(i, :) = basis_at_xp';
end


function [wt_dy] = get_HF_data_weights(xi, xHF, h_dist)
kerf = @(z) exp(-z.*z/2); % kernel function with max height of 1
[N_HF, ~] = size(xHF); 
% [N_HF, N_dim] = size(xHF); 
distx = sqrt(sum( (repmat(xi, N_HF, 1) - xHF).^2 , 2 ));
% margin_target = 10^(-min(N_HF^(1/N_dim)+3, 20)); % RANGES FROM A MIN OF 10^-20 TO A MAX OF 10^-4 (LESS SPARSE TO MORE SPARSE)
% scale_wt = sqrt(-2*log(margin_target*sqrt(2*pi))); % RANGES FROM 9.5 TO 4.07 (LESS SPARSE TO MORE SPARSE) [7 ~mean, 2.33 ratio]
% h = h_dist;%/scale_wt;
wt_dy = kerf(distx/h_dist); % GAUSSIAN WITH MIDDLE HEIGHT OF 1 (AREA VARIES WITH h_dist = SIG)
wt_dy = wt_dy/sum(wt_dy); % make weights sum to one (UNNECESSARY -- ONLY RELATIVE WEIGHTS MATTER | UNTRUE -- MATTERS FOR ERR)


function [C_wt, SINGULAR_ERR, COND_VAL] = ...
                        solve_localized_galerkin_at_point(yHF, basis_at_HF, wt_dy, SINGULAR_TOL)
% get number of functions
N_basis = length(basis_at_HF);
% Solve Localized Galerkin Matrix Equation
H_lgmf = zeros(N_basis, N_basis);
for i = 1:N_basis
    for j = 1:N_basis
        H_lgmf(i,j) = sum(wt_dy .* basis_at_HF{i} .* basis_at_HF{j});
    end
end
Y_lgmf = zeros(N_basis, 1);
for i=1:N_basis
    Y_lgmf(i) = sum(wt_dy .* basis_at_HF{i} .* yHF);
end
COND_VAL = cond(H_lgmf);
if COND_VAL > SINGULAR_TOL
    C_wt = 1/N_basis * ones(N_basis, 1);
    SINGULAR_ERR = true;
else
    C_wt = H_lgmf\Y_lgmf;
    SINGULAR_ERR = false;
end




function [X_design] = get_design_matrix(x_data, fit_type)
[N_pts, N_dim] = size(x_data); fit_type = lower(fit_type);
if strcmp(fit_type, 'constant')
    X_design = ones(N_pts, 1);
elseif strcmp(fit_type, 'linear')
    X_design = [ones(N_pts, 1), x_data];
elseif strcmp(fit_type, 'purequadratic')
    X_design = [ones(N_pts, 1), x_data, x_data.^2];
elseif strcmp(fit_type, 'interactions') || strcmp(fit_type, 'quadratic')
    N_fact = (N_dim^2 - N_dim)/2; % TODO: potentially get rid of this term
    x_interactions = zeros(N_pts, N_fact);
    col = 0;
    for j = 2:N_dim
        for i = 1:j-1
            col = col + 1;
            x_interactions(:, col) = x_data(:, i).*x_data(:, j);
        end
    end
    if col ~= N_fact %TODO: get rid of this (& col = 0) after confirming it works 
        error('creation of interaction terms needs work')
    end
    if strcmp(fit_type, 'interactions')
        X_design = [ones(N_pts, 1), x_data, x_interactions];
    else % must be 'quadratic'
        X_design = [ones(N_pts, 1), x_data, x_data.^2, x_interactions];
    end
else
    error(['The fit type of the design matrix must be one of "constant",'...
           ' "linear", "interactions", "purequadratic", and "quadratic"'])
end


