%% test the multigrid solver
%n = 64*16;
a = 0;
b = 1;
ua = 0;
ub = 0;

my_force = @force;
my_exact = @exact;
% [ u1, it_num1 ] = monogrid_poisson_1d ( n, a, b, ua, ub, my_force, my_exact );
% [ u2, it_num2 ] = multigrid_poisson_1d ( n, a, b, ua, ub, my_force, my_exact );
% 
% fprintf('Iteration number of monogrid is %d.\n',it_num1);
% plot(u1);
% hold on;
% 
% fprintf('Iteration number of multigrid is %d.\n',it_num2);
% plot(u2);

Nlist = [2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9];

for n=Nlist
    [ u1, it_num1 ] = monogrid_poisson_1d ( n, a, b, ua, ub, my_force, my_exact );
    fprintf('Iteration number of monogrid is %d.\n',it_num1);
%     [ u1, it_num1 ] = multigrid_poisson_1d ( n, a, b, ua, ub, my_force, my_exact );
%     fprintf('Iteration number of multigrid is %d.\n',it_num1);
    x1 = linspace(0,n,n+1)/n;
    plot(x1,u1,'DisplayName',append('N=',string(n)));
    hold on;
    legend
end

%saveas(gcf,'multigrid.png')
saveas(gcf,'monogrid.png')

% define the rhs function force(x)
function value = force(x)
    value = 1.*x; %x.^2;
end

% define the exact solution exact(x)
function value = exact(x)
    value = 0.*x;
end