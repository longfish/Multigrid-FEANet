function [uu] = u_exact

L = 2; H = 2; % domain size
Nx=64; Ny=64; % element number

para_C = 5;
para_k = 1;
para_l = 2;

[coordinates,~] = MeshRectanglularPlate(L,H,Nx,Ny);
uu = para_C/((pi*para_k)^2+(pi*para_l)^2)*sin(para_k*pi*coordinates(:,1)).*sin(para_l*pi*coordinates(:,2));

ii=0;
u = zeros(Nx+1);
for x = coordinates'
    xx = abs(x-[0;H]);
    ii = ii+1;
    j = int16(xx(1)/(L/Nx))+1;
    i = int16(xx(2)/(H/Ny))+1;
    u(i,j) = uu(ii);
end

% plot the exact solution
figure()
imagesc(u)
axis image
title('Exact solution')
colorbar

end