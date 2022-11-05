function [K, f, d, mesh] = Heat

T = 1; % total time
num_steps = 60;
dt = T/num_steps;
Di = 0.01; % diffusivity 
L = 1; H = 1; % domain size
Nx=36; Ny=36; 

%ElemSize = L/Nx;
[coordinates,nodes] = MeshRectanglularPlate(L,H,Nx,Ny);
mesh.x=coordinates';
mesh.conn=nodes';

ElementNodesNum=4; %Number of the element nodes
ElementDof=ElementNodesNum; % Dof of the element
GDof=size(mesh.x,2); %Global Dof
%ElementNum=size(mesh.conn,2);

mesh.qpts = [-1 1 1 -1; -1 -1 1 1]/sqrt(3); %quadrature poins, weights are 1s

% Compute stiffness matrix and external forces
% K=sparse(GDof,GDof);
% f=sparse(GDof,1);
K=zeros(GDof,GDof);

i = 0;
%Assembly of stiffness matrix and body force
for c = mesh.conn
    xe = mesh.x(:,c); % element node coordinate
    Ke = zeros(ElementDof);
    for q = mesh.qpts 
        [N, dNdp] = shape(q);
        J = xe * dNdp;
        dNdx = dNdp/J;
        B = zeros(2,ElementDof); % 2 rows for heat problem
        B(1,:) = dNdx(:,1)';
        B(2,:) = dNdx(:,2)';
        D=Di*eye(2);
        %Ke = Ke + (N * N' + dt * B' * D * B);
        %Ke = Ke + (N * N' + dt * B' * D * B) * det(J);
        %Ke = Ke + (N * N'*det(J)) - dt * dNdp * D * dNdp';
        %Ke = Ke + N * N' * det(J) + dt * B' * D * B;
        %Ke = Ke + dt *B' * D * B * det(J);
        Ke = Ke + B' * D * B * det(J);
    end
    Sctr = c;
    K(Sctr,Sctr) = K(Sctr,Sctr) + Ke;
    i=i+1;
    if i==2
        Ke
    end
end

% Apply essential/displacement boundary condition 
% According to FEM book, use the partition approach (p.g. 20)
mesh.node_list = linspace(1,length(coordinates),length(coordinates));
mesh.node_upper = find(mesh.x(2,:) == H);
mesh.node_bottom = find(mesh.x(2,:) == 0);
mesh.node_left = find(mesh.x(1,:) == 0);
mesh.node_right = find(mesh.x(1,:) == L);
mesh.node_essential = unique([mesh.node_upper,mesh.node_bottom,mesh.node_left,mesh.node_right]);

% set essential bc for upper nodes
d = zeros(length(coordinates),1);
d(mesh.node_upper) = 1.0; 
d_E = d(mesh.node_essential);

%K_E = K(node_essential,node_essential);
K_EF = K(mesh.node_essential,setdiff(mesh.node_list,mesh.node_essential));
K_F = K(setdiff(mesh.node_list,mesh.node_essential),setdiff(mesh.node_list,mesh.node_essential));

% update the solution during t=[0,T]
f = zeros(GDof,1); 
for t = 1:1
    fprintf("Step-%d\n",t);
    f = f_update(mesh,d);
    f_F = f(setdiff(mesh.node_list,mesh.node_essential));
    d_F = K_F\(f_F - K_EF' * d_E);
    d(setdiff(mesh.node_list,mesh.node_essential)) = d_F;
end

% save 2D data
ii=0;
tt = zeros(Nx+1);
for x = mesh.x
    xx = abs(x-[0;H]);
    ii = ii+1;
    j = int16(xx(1)/(L/Nx))+1;
    i = int16(xx(2)/(H/Ny))+1;
    tt(i,j) = d(ii);
end

save('heat_36x36_xy_step1.mat','tt');

vtkwrite('heat_36x36_xy_step1.vtk', 'structured_points', 'T', tt);

% plot the temperature distributions
figure()
imagesc(tt)
axis image
title('tt')
colorbar

% nicer method
%patch('vertices',mesh.x','faces',mesh.conn','facecolor','interp',...
%    'facevertexcdata',d);
%axis image

end


function [N, dNdp] = shape(p)
% shape function
N = 0.25*[(1-p(1))*(1-p(2));
    (1+p(1))*(1-p(2));
    (1+p(1))*(1+p(2));
    (1-p(1))*(1+p(2))];

dNdp = 0.25*[-(1-p(2)), -(1-p(1));
    (1-p(2)), -(1+p(1));
    (1+p(2)), (1+p(1));
    -(1+p(2)), (1-p(1))];
end

function f_o = f_update(mesh,u)
% update f using u
f_o = zeros(length(u),1);
for c = mesh.conn
    xe = mesh.x(:,c); % element node coordinate
    fe = zeros(length(mesh.qpts),1);
    temp = zeros(4);
    for q = mesh.qpts 
        [N, dNdp] = shape(q);
        J = xe * dNdp;
        temp = temp + N*N'*det(J);
        fe = fe+N*N'*det(J)*u(c);
    end
    f_o(c) = f_o(c) + fe;
end
end

