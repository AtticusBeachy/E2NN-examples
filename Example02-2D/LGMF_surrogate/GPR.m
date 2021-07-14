classdef GPR
    properties
        x_min
        x_max
        y_min
        y_max
        x
        y
        x_norm
        y_norm
        theta
        dmodel_norm
    end
    methods
        function obj = GPR(X,Y,theta0,regr)
            
            % --- eliminate duplicate data
            x_unq = []; % X with duplicates removed
            y_unq = []; % Y with x-duplicates removed
            for i = 1:size(X, 1) % current point
                add_point = true;
                for j = 1:i-1 % previous points
                    differences = abs( X(j,:) - X(i,:) );
                    if all(differences < 10^-5) % if points overlap
                        add_point = false;
                    end
                end
                if add_point
                    x_unq = [x_unq; X(i,:)];
                    y_unq = [y_unq; Y(i,:)];
                end
            end
            
            
            obj.x = x_unq; obj.y = y_unq;
            [obj.x_norm, obj.x_min, obj.x_max] = NormalizeData(x_unq);
            [obj.y_norm, obj.y_min, obj.y_max] = NormalizeData(y_unq);
            
%             regr = @regpoly0; % first order regression (0, 1, 2)
            corr = @corrgauss;
            dim = size(x_unq,2);
            
%             lob = 0.1 * ones(dim, 1);
%             upb = 20 * ones(dim, 1);
            lob = 10^-6 * ones(dim, 1);
            upb = 100 * ones(dim, 1);
            
            if ~exist('regr', 'var') || isempty(regr)
                regr = @regpoly0;
            end
            
            if ~exist('theta0', 'var') || isempty(theta0)
                theta0 = 10 * ones(dim, 1);
                dace_model_norm = dacefit_robust(obj.x_norm, obj.y_norm, regr, corr, theta0, lob, upb);
            else
                dace_model_norm = dacefit_update(obj.x_norm, obj.y_norm, regr, corr, theta0, lob, upb);
%                 dace_model_norm = dacefit_robust(obj.x_norm, obj.y_norm, regr, corr, theta0, lob, upb);
            end
            
            obj.theta = dace_model_norm.theta; %(optimized theta)
            obj.dmodel_norm = dace_model_norm;
        end
        
        function [yp, sig_yp] = predictGPR(obj, xp)
            
            Ndata = size(xp, 1);
            yp_norm = zeros(Ndata, 1);
            ypMSE = zeros(Ndata, 1);
            
            % Memory issues cause a crash if xp has too many elements
            % capping at 1000
            
            Nsubset = 1000;
            start_idx = 1:Nsubset:Ndata;
            end_idx = [start_idx(2:end)-1, Ndata];
            
            for b = 1:length(start_idx)
                
                idxs = start_idx(b):end_idx(b);
                xp_sub = xp(idxs,:);
                
                % the prediction fails if xp only contains one point.
                % correcting this:
                if size(xp_sub, 1) == 1
                    xp_sub = repmat(xp_sub, 2, 1);
                    reduce_dim = true;
                else
                    reduce_dim = false;
                end
                
                xp_norm = NormalizeData(xp_sub, obj.x_min, obj.x_max);
                
                [yp_norm_sub, ypMSE_sub] = predictor(xp_norm, obj.dmodel_norm);
                
                
                % finish correction if single point
                if reduce_dim
                    yp_norm_sub = yp_norm_sub(1,:);
                    ypMSE_sub = ypMSE_sub(1,:);
                end
                
                % store subset of results
                yp_norm(idxs,:) = yp_norm_sub;
                ypMSE(idxs,:) = ypMSE_sub;
                
            end
            
            y_rmse_n = sqrt(ypMSE);
            yp = DeNormMinMax( yp_norm, obj.y_min, obj.y_max );
            %sig_yp = DeNormMinMax( y_rmse_n, obj.y_min, obj.y_max );
            sig_yp = (obj.y_max - obj.y_min)*y_rmse_n;
            
            
        end
    end
end