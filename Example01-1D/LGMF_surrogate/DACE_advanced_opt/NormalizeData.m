function [ Xnorm,Xmin,Xmax ] = NormalizeData( X, Xmin, Xmax )
% This function normalizes a maxtrix of x-data
% The x-values are column vectors 
% (each column specifies a different dimension)
% the data is normalized from Xmin = 0 to Xmax = 1. If Xmin and Xmax are
% not included, the minimum and maximum values in X are used.

[row, col] = size(X);
Xnorm = zeros(row, col);

if ~exist('Xmin','Var') || isempty(Xmin)
    Xmin = min(X,[],1);
    Xmax = max(X,[],1);
end

for i = 1:col
    Xnorm(:,i) = (X(:,i) - Xmin(i))/( Xmax(i) - Xmin(i));
end

end

