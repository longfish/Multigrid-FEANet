function [mesh] = hole_mesh(nh, L, rx, ry)
% MAKE_MESH Returns a mesh for a square plate with a center hole.
%   Input arguments: nh - number of elements along half of a side.
%                    side_length - length of a side of the plate.
%                    rx - radius along x direction of hole.
%                    ry - radius along x direction of hole.
    n_shell = 2*nh;
    nns = 8*nh;  % number of nodes in a shell.            
    circle = make_circle(nns, rx, ry)';
    square = L*make_square(nh)';
    mesh.x = [];
    for s = 1:n_shell
        % f goes from 0 to 1 as a linear function.
        f = (s-1)/(n_shell-1);
        mesh.x = [mesh.x, square*(1-f) + circle*f];
    end    
    
    mesh.conn = [];
    for j=1:n_shell-1
        for i=1:nns-1               
          n0 = i + (j-1)*nns;          
          mesh.conn(:,end+1) = [n0; n0+1; n0+1+nns; n0+nns];           
       end               
       mesh.conn(:,end+1) = [1+(j-1)*nns; 1+(j)*nns; 
                             mesh.conn(3,end); mesh.conn(2,end)];
    end
    mesh.bottom_nodes = find(mesh.x(2,:) == 0);
    mesh.top_nodes = find(mesh.x(2,:) == L);
    mesh.left_nodes = find(mesh.x(1,:) == 0);
    mesh.right_nodes = find(mesh.x(1,:) == L);
    mesh.circle_nodes = length(mesh.x)-nns+1:length(mesh.x);
    
    fprintf('Created mesh with %d elements, and %d nodes.\n', ...
        length(mesh.conn), length(mesh.x));
end

function [x] = make_square(nh)    
    x(1:nh+1,1) = 0.5;
    x(1:nh+1,2) = linspace(0, 0.5, nh+1);
    x(nh+1:3*nh+1,1) = linspace(0.5, -0.5, 2*nh+1);
    x(nh+1:3*nh+1,2) = 0.5;    
    x(3*nh+1:5*nh+1,1) = -0.5;
    x(3*nh+1:5*nh+1,2) = linspace(0.5, -0.5, 2*nh+1);    
    x(5*nh+1:7*nh+1,1) = linspace(-0.5, 0.5, 2*nh+1);
    x(5*nh+1:7*nh+1,2) = -0.5;
    x(7*nh+1:8*nh,1) = 0.5;    
    x(7*nh+1:8*nh,2) = linspace(-0.5,-0.5/nh,nh);
end

function [x] = make_circle(nns, rx, ry)
    q = linspace(0, 2*pi, nns+1)';
    x = [rx*cos(q(1:end-1)), ry*sin(q(1:end-1))];
end
