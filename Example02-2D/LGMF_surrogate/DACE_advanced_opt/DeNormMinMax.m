function [ notNorm ] = DeNormMinMax( Xnorm, MIN, MAX )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[~,col] = size(Xnorm);
notNorm = Xnorm;

for i = 1:col
    notNorm(:,i) = Xnorm(:,i).*( MAX(i) - MIN(i)) + MIN(i); 
end

end
