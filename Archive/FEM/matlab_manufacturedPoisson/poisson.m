function [K, f_o, f, d, mesh, error_arr] = poisson

L = 2; H = 2; % domain size
Nx=64; Ny=64; % element number

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

%Assembly of stiffness matrix and body force
for c = mesh.conn
    xe = mesh.x(:,c); % element node coordinate
%     centroid = mean(xe,2); % centroid of the element
%     if (centroid(1)-L/2)^2+(centroid(2)-H/2)^2 < 0.5*0.5
%         Di = 20;
%     else
%         Di = 1;
%     end
    Di = 1;
    Ke = zeros(ElementDof);
    for q = mesh.qpts 
        [~, dNdp] = shape(q);
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
d(mesh.node_upper) = 0.0; 
d_E = d(mesh.node_essential);

%K_E = K(node_essential,node_essential);
K_EF = K(mesh.node_essential,setdiff(mesh.node_list,mesh.node_essential));
K_F = K(setdiff(mesh.node_list,mesh.node_essential),setdiff(mesh.node_list,mesh.node_essential));

% construct the rhs
para_C = 5;
para_k = 1;
para_l = 2;
f = para_C*sin(para_k*pi*coordinates(:,1)).*sin(para_l*pi*coordinates(:,2));
%f = ones(GDof,1); 
f_o = f_update(mesh,f);

f = f_update(mesh,f);
f_F = f(setdiff(mesh.node_list,mesh.node_essential));
%d_F = K_F\(f_F - K_EF' * d_E);
% solve using Jacobi iteration
A = K_F;
b = f_F - K_EF' * d_E;

n_iter = 800;
[d_F, error_arr] = jacobi_method(A, b, zeros(length(b),1), n_iter);
d(setdiff(mesh.node_list,mesh.node_essential)) = d_F;


% % compute the residual
% uu = u_exact();
% ff = K*uu;
% residual = ff - f_o;

% save 2D data
ii=0;
u = zeros(Nx+1);
% rr = zeros(Nx+1);
for x = mesh.x
    xx = abs(x-[0;H]);
    ii = ii+1;
    j = int16(xx(1)/(L/Nx))+1;
    i = int16(xx(2)/(H/Ny))+1;
    u(i,j) = d(ii);
    % rr(i,j) = residual(ii);
end

save('poisson_xy.mat','u','error_arr');

vtkwrite('poisson_xy.vtk', 'structured_points', 'u', u);

% plot the temperature distributions
figure()
imagesc(u)
axis image
title('Solution')
colorbar

% plot the error 
figure()
plot(error_arr)
ylim([0,0.5])
title('Error')


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

function f_o = f_update(mesh,ff)
% update f using source term ff
f_o = zeros(length(ff),1);
for c = mesh.conn
    xe = mesh.x(:,c); % element node coordinate
    fe = zeros(length(mesh.qpts),1);
    temp = zeros(4);
    for q = mesh.qpts 
        [N, dNdp] = shape(q);
        J = xe * dNdp;
        temp = temp + N*N'*det(J);
        fe = fe+N*N'*det(J)*ff(c);
    end
    f_o(c) = f_o(c) + fe;
end
end

function [x, error_arr] = jacobi_method(A, b, x0, iteration)
% output the solution x and relative error between two solutions
Dinv = diag(1./diag(A));
N = 1;
x = x0;
error_arr = zeros(iteration+1,1);error_arr(1) = 1;
omega = 2/3; % weighted Jacobi iteration
while(N<=iteration)
    xprev = x;
    residual = b - A*xprev;
    x = omega*Dinv*residual + xprev;
    N = N+1;
    error_arr(N) = norm(x-xprev)/norm(x);
end
end