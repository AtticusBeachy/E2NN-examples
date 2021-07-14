function  [dmodel, perf] = dacefit_bh01(S, Y, regr, corr, theta0, lob, upb)
%DACEFIT Constrained non-linear least-squares fit of a given correlation
% model to the provided data set and regression model
%
% Call
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0)
%   [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
%
% Input
% S, Y    : Data points (S(i,:), Y(i,:)), i = 1,...,m
% regr    : Function handle to a regression model
% corr    : Function handle to a correlation function
% theta0  : Initial guess on theta, the correlation function parameters
% lob,upb : If present, then lower and upper bounds on theta
%           Otherwise, theta0 is used for theta
%
% Output
% dmodel  : DACE model: a struct with the elements
%    regr   : function handle to the regression model
%    corr   : function handle to the correlation function
%    theta  : correlation function parameters
%    beta   : generalized least squares estimate
%    gamma  : correlation factors
%    sigma2 : maximum likelihood estimate of the process variance
%    S      : scaled design sites
%    Ssc    : scaling factors for design arguments
%    Ysc    : scaling factors for design ordinates
%    C      : Cholesky factor of correlation matrix
%    Ft     : Decorrelated regression matrix
%    G      : From QR factorization: Ft = Q*G' .
% perf    : struct with performance information. Elements
%    nv     : Number of evaluations of objective function
%    perf   : (q+2)*nv array, where q is the number of elements 
%             in theta, and the columns hold current values of
%                 [theta;  psi(theta);  type]
%             |type| = 1, 2 or 3, indicate 'start', 'explore' or 'move'
%             A negative value for type indicates an uphill step

% hbn@imm.dtu.dk  
% Last update September 3, 2002

% Check design points
[m n] = size(S);  % number of design sites and their dimension
sY = size(Y);
if  min(sY) == 1,  Y = Y(:);   lY = max(sY);  sY = size(Y);
else,              lY = sY(1); end
if m ~= lY
  error('S and Y must have the same number of rows'), end

% Check correlation parameters
lth = length(theta0);
if  nargin > 5  % optimization case
  if  length(lob) ~= lth | length(upb) ~= lth
    error('theta0, lob and upb must have the same length'), end
  if  any(lob <= 0) | any(upb < lob)
    error('The bounds must satisfy  0 < lob <= upb'), end
else  % given theta
  if  any(theta0 <= 0)
    error('theta0 must be strictly positive'), end
end

% Normalize data
mS = mean(S);   sS = std(S);
mY = mean(Y);   sY = std(Y);

minS=min(S);
maxS=max(S);
mS=minS;
sS=maxS-minS;
minY=min(Y);
maxY=max(Y);
mY=minY
sY=maxY-minY

% 02.08.27: Check for 'missing dimension'
j = find(sS == 0);
if  ~isempty(j),  sS(j) = 1; end
j = find(sY == 0);
if  ~isempty(j),  sY(j) = 1; end

original_S=S
original_Y=Y

S = (S - repmat(mS,m,1)) ./ repmat(sS,m,1);
Y = (Y - repmat(mY,m,1)) ./ repmat(sY,m,1);

% Calculate distances D between points
mzmax = m*(m-1) / 2;        % number of non-zero distances
ij = zeros(mzmax, 2);       % initialize matrix with indices
D = zeros(mzmax, n);        % initialize matrix with distances
ll = 0;
for k = 1 : m-1
  ll = ll(end) + (1 : m-k);
  ij(ll,:) = [repmat(k, m-k, 1) (k+1 : m)']; % indices for sparse matrix
  D(ll,:) = repmat(S(k,:), m-k, 1) - S(k+1:m,:); % differences between points
end
if  min(sum(abs(D),2) ) == 0
  error('Multiple design sites are not allowed'), end

% Regression matrix
F = feval(regr, S);  [mF p] = size(F);
if  mF ~= m, error('number of rows in  F  and  S  do not match'), end
if  p > mF,  error('least squares problem is underdetermined'), end

% parameters for objective function
par = struct('corr',corr, 'regr',regr, 'y',Y, 'F',F, ...
  'D', D, 'ij',ij, 'scS',sS);



% Determine theta

max_D=max(sqrt(sum(par.D.^2,2)))
%theta_upper_limit that covers 98% correlation
upb=-log(0.02)/max_D^2


theta_search=[];
save theta_search theta_search


if  nargin > 5
  % Bound constrained non-linear optimization
  [theta f fit perf] = boxmin(theta0, lob, upb, par);
  if  isinf(f)
    error('Bad parameter region.  Try increasing  upb'), end
else
  % Given theta
  theta = theta0(:);   
  [f  fit] = objfunc(theta, par);
  perf = struct('perf',[theta; f; 1], 'nv',1);
  if  isinf(f)
    error('Bad point.  Try increasing theta0'), end
end

% Return values
dmodel = struct('regr',regr, 'corr',corr, 'theta',theta.', ...
  'beta',fit.beta, 'gamma',fit.gamma, 'sigma2',sY.^2.*fit.sigma2, ...
  'S',S, 'Ssc',[mS; sS], 'Ysc',[mY; sY], ...
  'C',fit.C, 'Ft',fit.Ft, 'G',fit.G);

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================

    
% --------------------------------------------------------    
function t0=starting_point(lo,up,par)



%%%%%%%%%%%%%%%%%%%%%%%%% bh only for 2 params
tests=[1:5]'

% 5 levels
lv=length(tests);

db=(up-lo)/(lv-1);
f=[];

for i=1:length(tests(:,1))
    testi=tests(i,:);
    testix=lo+(testi-1).*db;
    fi=objfunc(testix, par);
    f=[f fi];
end

[min_f,index]=min(f);
t0=lo+(tests(index,:)-1).*db;


    
% --------------------------------------------------------

function  [t_opt, f, fit, perf] = boxmin(t0, lo, up, par)
%BOXMIN  Minimize with positive box constraints

t0=starting_point(lo,up,par)



options=optimset('TolFun',1e-18,'TolX',1e-12,'Display','on')
[t_opt,fval,exitflag] = ... 
   fmincon(@objfunc,t0,[],[],[],[],lo,up,[],options,par)


t_opt
[f, fit] = objfunc(t_opt, par)
perf = f


% --------------------------------------------------------

function  [obj, fit] = objfunc(theta, par)
% Initialize
obj = inf; 
fit = struct('sigma2',NaN, 'beta',NaN, 'gamma',NaN, ...
    'C',NaN, 'Ft',NaN, 'G',NaN);
m = size(par.F,1);
% Set up  R
r = feval(par.corr, theta, par.D);
idx = find(r > 0);   o = (1 : m)';   
mu = (10+m)*eps;
R = sparse([par.ij(idx,1); o], [par.ij(idx,2); o], ...
  [r(idx); ones(m,1)+mu]);  
% Cholesky factorization with check for pos. def.
[C rd] = chol(R);
if  rd,  return, end % not positive definite

% Get least squares solution
C = C';   Ft = C \ par.F;
[Q G] = qr(Ft,0);
if  rcond(G) < 1e-10
  % Check   F  
  if  cond(par.F) > 1e15 
    T = sprintf('F is too ill conditioned\nPoor combination of regression model and design sites');
    error(T)
  else  % Matrix  Ft  is too ill conditioned
    return 
  end 
end
Yt = C \ par.y;   beta = G \ (Q'*Yt);
rho = Yt - Ft*beta;  sigma2 = sum(rho.^2)/m;
detR = prod( full(diag(C)) .^ (2/m) );
obj = sum(sigma2) * detR;


if  nargout > 1
  fit = struct('sigma2',sigma2, 'beta',beta, 'gamma',rho' / C, ...
    'C',C, 'Ft',Ft, 'G',G');
end


load theta_search
theta_search=[theta_search;theta obj]
save theta_search theta_search



