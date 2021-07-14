function ex01_1D01

load temp-2d-compared-to-dace_sN40_rstd10p.mat
S=X;

 theta=[1.9953e+01   1.9953e+01];
 lob=[0.1 0.1];
 upb=[30 30]*2;
 
 [dmodel, perf]=dacefit_bh01(S,Y,@regpoly2, @corrgauss, theta, lob,upb);
%   [dmodel, perf]=dacefit(S,Y,@regpoly0, @corrgauss, theta, lob,upb);
 


 

 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  prediction samples [start]

plotds=[1 2];
xmins=min(S);
xmaxs=max(S);

xmins=xmins(plotds);
xmaxs=xmaxs(plotds);

gridn=20;
dxs=(xmaxs-xmins)/gridn;
[x1,x2]=meshgrid(xmins(1):dxs(1):xmaxs(1),xmins(2):dxs(2):xmaxs(2));
ys=[];
MSEs=[];

for i=1:length(x1(:,1))
    for j=1:length(x2(:,1))
        cx=[x1(i,j) x2(i,j)]';
        [ysi,or1,MSE1]=predictor(cx, dmodel) ;
        ys(i,j)=ysi;
        MSEs(i,j)=sqrt(abs(MSE1));
    end
end


figure    
scatter3(X(:,1),X(:,2),Y,'filled')
hold on
surf(x1,x2,ys)
surf(x1,x2,ys+3*MSEs,'FaceAlpha',0.3)
surf(x1,x2,ys-3*MSEs,'FaceAlpha',0.3)
axis([0 1 0 1 -2 2])
view([120 20])
xlabel('x1')
ylabel('x2')
zlabel('y')
axis vis3d  
MSEs;

