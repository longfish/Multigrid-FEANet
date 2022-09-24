function Plane_Stress_modify

L = 0.2; H = 0.2;
Nx=36; Ny=36; 
ElemSize = L/Nx;
[coordinates,nodes] = MeshRectanglularPlate(L,H,Nx,Ny);

mesh.x=coordinates';
mesh.conn=nodes';

E0=[212e9,212e9];
poisson0=[0.288,0.288]; 
 

ElementNodesNum=4; %Number of the element nodes
ElementDof=2*ElementNodesNum; % Dof of the element
%thickness=0.01; % Thickness
GDof=2*size(mesh.x,2); %Global Dof
%ElementNum=size(mesh.conn,2);

% Bottomedge_conn = [mesh.bottom_nodes(1:end-1); mesh.bottom_nodes(2:end)];
% RVariableNum=2*ElementNum+size(Bottomedge_conn,2);
% RVariableDof=zeros(RVariableNum); %Initial of the random variables (ro,E,traction)
% bbar=[0 ;-9.8*ro*thickness]; % Body force (gravity) (x,y)'
% bbar=zeros(2,ElementNum);

% Lower=[20,10,5,1,100];
% Upper=[60,100,15,6,200];
% Num_simu=2;
%  for i=1:5
%      a(i,:)=unifrnd(Lower(i),Upper(i),1,Num_simu);
%  end

a = zeros(5,1);

% grad_D=1/(1-poisson^2)*[1 poisson 0;poisson 1 0;0 0 (1-poisson)/2]; %Strain-Stress Relationship for Plane Stress

qpts = [-1 1 1 -1; -1 -1 1 1]/sqrt(3); %quadrature poins, intergral points
% Compute stiffness matrix and external forces
% K=sparse(GDof,GDof);
K=zeros(GDof,GDof);
f=zeros(GDof,1); %External nodes force
% f=sparse(GDof,1);
%Assembly of stiffness matrix and body force

% i=0;
% Steel_conn=[];
for c = mesh.conn
    
    xe = mesh.x(:,c);
    xee1=mean(xe,2);
    xee=xee1-[0.005*33;0.005*33];
    if  sum(xee.^2)^0.5<16*0.005   % determine the radius of the inner material
        poisson=poisson0(1);
        E=E0(1);
        grad_D=1/(1-poisson^2)*[1 poisson 0;poisson 1 0;0 0 (1-poisson)/2]; %Strain-Stress Relationship for Plane Stress
      
%        XE(:,end+1)=xee1;
%        C(end+1:end+4,1)=c;
    else 
%        Steel_conn(:,end+1) = c;
        
        E=E0(2);
        poisson=poisson0(2);
        grad_D=1/(1-poisson^2)*[1 poisson 0;poisson 1 0;0 0 (1-poisson)/2]; %Strain-Stress Relationship for Plane Stress
    end
    Ke = zeros(ElementDof);
    fe = zeros(ElementDof,1);
    for q = qpts 
        [N, dNdp] = shape(q);
        J = xe * dNdp;
        dNdx = dNdp/J;
        B = zeros(3,ElementDof);
        B(1,1:2:8) = dNdx(:,1)';
        B(2,2:2:8) = dNdx(:,2)';
        B(3,1:2:8) = dNdx(:,2)';
        B(3,2:2:8) = dNdx(:,1)';
        D=E*grad_D;
        Ke = Ke + B' * D * B * det(J);
        be=bbar(xe,a);
        fe(1:2:end,1)=fe(1:2:end,1)+N*N'*be(1:2:end,1);
        fe(2:2:end,1)=fe(2:2:end,1)+N*N'*be(2:2:end,1);
    end
    Sctr(1:2:ElementDof) = 2*c-1;
    Sctr(2:2:ElementDof) = 2*c;
    K(Sctr,Sctr) = K(Sctr,Sctr) + Ke;
    f(Sctr) = f(Sctr) + fe;
end

% Apply traction force along X_direction

% find middle five elements
mesh.middle_nodes = find(mesh.x(1,:) >= L/2-ElemSize & mesh.x(1,:)<= L/2+ElemSize & ...
    mesh.x(2,:) >= H/2-3*ElemSize & mesh.x(2,:)<= H/2+3*ElemSize);
% loading along x direction
BC_right_nodes_Sctr=2*mesh.middle_nodes-1;
f(BC_right_nodes_Sctr) = 100000;

% find rhs five elements
mesh.middle_nodes = find(mesh.x(2,:) >= L/2-ElemSize & mesh.x(2,:)<= L/2+ElemSize & ...
    mesh.x(1,:) >= H/2-3*ElemSize & mesh.x(1,:)<= H/2+3*ElemSize);
% loading along y direction
BC_right_nodes_Sctr=2*mesh.middle_nodes;
f(BC_right_nodes_Sctr) = 100000;

% j=0;
% for c = rightedge_conn
%     j = j + 1;
%     xe = mesh.x(:, c);
%     le = norm(xe(:,2) - xe(:,1));
%     N = [0.5; 0.5];
%     Traction_Sctr = 2*c-1; %X_direction
%     f(Traction_Sctr) = f(Traction_Sctr) - N * (tbar(xe(1,2))+tbar(xe(2,2)))/2  * le; % traction on the right edge
% end



%Apply essential/displacement boundary condition 

% top
mesh.node_upper = find(mesh.x(2,:) == H);
BC_node_upper_Sctr_x=2*mesh.node_upper-1;
BC_node_upper_Sctr_y=2*mesh.node_upper;
K(BC_node_upper_Sctr_y,:) = 0;
K(:,BC_node_upper_Sctr_y) = 0;
K(BC_node_upper_Sctr_y,BC_node_upper_Sctr_y) = eye(length(BC_node_upper_Sctr_y));
f(BC_node_upper_Sctr_y) = 0;
K(BC_node_upper_Sctr_x,:) = 0;
K(:,BC_node_upper_Sctr_x) = 0;
K(BC_node_upper_Sctr_x,BC_node_upper_Sctr_x) = eye(length(BC_node_upper_Sctr_x));
f(BC_node_upper_Sctr_x) = 0;

% bottom
mesh.node_bottom = find(mesh.x(2,:) == 0);
BC_node_bottom_Sctr_x=2*mesh.node_bottom-1;
BC_node_bottom_Sctr_y=2*mesh.node_bottom;
K(BC_node_bottom_Sctr_y,:) = 0;
K(:,BC_node_bottom_Sctr_y) = 0;
K(BC_node_bottom_Sctr_y,BC_node_bottom_Sctr_y) = eye(length(BC_node_bottom_Sctr_y));
f(BC_node_bottom_Sctr_y) = 0;
K(BC_node_bottom_Sctr_x,:) = 0;
K(:,BC_node_bottom_Sctr_x) = 0;
K(BC_node_bottom_Sctr_x,BC_node_bottom_Sctr_x) = eye(length(BC_node_bottom_Sctr_x));
f(BC_node_bottom_Sctr_x) = 0;

% left
mesh.node_upper = find(mesh.x(1,:) == 0);
BC_node_upper_Sctr_x=2*mesh.node_upper-1;
BC_node_upper_Sctr_y=2*mesh.node_upper;
K(BC_node_upper_Sctr_y,:) = 0;
K(:,BC_node_upper_Sctr_y) = 0;
K(BC_node_upper_Sctr_y,BC_node_upper_Sctr_y) = eye(length(BC_node_upper_Sctr_y));
f(BC_node_upper_Sctr_y) = 0;
K(BC_node_upper_Sctr_x,:) = 0;
K(:,BC_node_upper_Sctr_x) = 0;
K(BC_node_upper_Sctr_x,BC_node_upper_Sctr_x) = eye(length(BC_node_upper_Sctr_x));
f(BC_node_upper_Sctr_x) = 0;

% right
mesh.node_upper = find(mesh.x(1,:) == L);
BC_node_upper_Sctr_x=2*mesh.node_upper-1;
BC_node_upper_Sctr_y=2*mesh.node_upper;
K(BC_node_upper_Sctr_y,:) = 0;
K(:,BC_node_upper_Sctr_y) = 0;
K(BC_node_upper_Sctr_y,BC_node_upper_Sctr_y) = eye(length(BC_node_upper_Sctr_y));
f(BC_node_upper_Sctr_y) = 0;
K(BC_node_upper_Sctr_x,:) = 0;
K(:,BC_node_upper_Sctr_x) = 0;
K(BC_node_upper_Sctr_x,BC_node_upper_Sctr_x) = eye(length(BC_node_upper_Sctr_x));
f(BC_node_upper_Sctr_x) = 0;

% BC1 = find(mesh.x(1,:) == 0 & mesh.x(2,:) == 0);
% BC1_Sctr = 2*BC1-1;
% f(BC1_Sctr) = 0;
% 
% BC2 = find(mesh.x(1,:) == L & mesh.x(2,:) == 0);
% BC2_Sctr = 2*BC2-1;
% f(BC2_Sctr) = 0;
% 
% BC3 = find(mesh.x(1,:) == L & mesh.x(2,:) == H);
% BC3_Sctr = 2*BC3-1;
% f(BC3_Sctr) = 0;
% 
% BC4 = find(mesh.x(1,:) == 0 & mesh.x(2,:) == H);
% BC4_Sctr = 2*BC4-1;
% f(BC4_Sctr) = 0;


% KInverse=K\eye(size(K));
% d=KInverse*f;  %displacements of the nodes
d=K\f;
% d(Steel_conn,1)=dd;
d_x = d(1:2:end,1);
d_y = d(2:2:end,1);

f_x = f(1:2:end,1);  
f_y = f(2:2:end,1);

% save 2D data
ii=0;
 for x = mesh.x
     x = abs(x-[0;H]);
     ii = ii+1;
     j = int16(x(1)/(L/Nx))+1;
     i = int16(x(2)/(H/Ny))+1;
     ux(i,j) = d_x(ii);
     fx(i,j) = f_x(ii);
     uy(i,j) = d_y(ii);
     fy(i,j) = f_y(ii);
     utem(i,j) = 0;
     ftem(i,j) = 0;
 end

ftem(12,12) = 1;

save('center_crack_36x36_xy.mat','ux','uy','utem','fx','fy','ftem');
%save('center_crack_36x36_xy.mat','ux','uy','fx','fy');

% plot the displacement and loading distributions
figure(1)
imagesc(fx)
axis image
title('Fx')

figure(2)
imagesc(fy)
axis image
title('Fy')

figure(3)
imagesc(ux)
axis image
title('ux')

figure(4)
imagesc(uy)
axis image
title('uy')

end





function [N, dNdp] = shape(p)
% shape function
N = 0.25*[(1-p(1)).*(1-p(2));
    (1+p(1)).*(1-p(2));
    (1+p(1)).*(1+p(2));
    (1-p(1)).*(1+p(2))];

dNdp = 0.25*[-(1-p(2)), -(1-p(1));
    (1-p(2)), -(1+p(1));
    (1+p(2)), (1+p(1));
    -(1+p(2)), (1-p(1))];
end

function f=tbar(x)
f=400e6;
end


function f=bbar(x,u)
f=[];
for i=1:size(x,2);
k=u(1,:);
a=u(2,:);
b=u(3,:);
c=u(4,:);
d=u(5,:);
% f= k*(7*sin(a*x(1,:)+b*x(2,:))+210*exp(c*x(1))+x(1)*x(2).^2)+d;
ff =k.*(sin(a*x(1,i)+b*x(2,i))+exp(c*x(1,i))+x(1,i)*x(2,i).^2)+d;
f(end+1:end+2,1)=1e3*ff';
f(:,:)=0;
end
end
