% mesh1 = hole_mesh(5,0.5,0.2,0.1);
% mesh1 = hole_mesh(5,1,0,0);
% mesh.x=mesh1.x;
% mesh.conn=mesh1.conn;

p.vertices = mesh.x';
p.faces = mesh.conn';
p.facecolor = 'w';
p.EdgeColor='b';
p.LineWidth=0.01;
patch(p);
axis square;
hold on;
scatter(mesh.x(1,:), mesh.x(2,:), 'filled');

% for i = 1:length(mesh.x)
%     text(mesh.x(1,i)+0.025, mesh.x(2,i), sprintf('%d', i));
% end



figure
mesh.x1=mesh.x+100*[d(1:2:end,1)';d(2:2:end,1)'];
p.vertices = mesh.x1';
p.faces = mesh.conn';
p.facecolor = 'w';
p.EdgeColor='b';
p.LineWidth=0.01;
patch(p);
axis square;
hold on;
scatter(mesh.x1(1,:), mesh.x1(2,:), 'filled');