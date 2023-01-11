function [K] = solve_plane_strain()%coor, conn, E, v)
    
    ey = 4;
    L = 3;
    r = 0.2;   %r_c = 0.23646556429...  critical radius
    E = 69e9;
    v = 0.34;

    mesh = beam_mesh(ey,L,r);
%     scatter(mesh.x(1,:),mesh.x(2,:));
    coor = mesh.x;
    conn = mesh.conn;


    %plane strain condition
    Ce = [1-v,  v ,   0;
           v,  1-v,   0;
           0,   0 , 0.5-v] * E/(1+v)/(1-2*v);
% Ce = E/(1-v^2)*[1 v 0;
%                v 1 0
%                0 0 (1-v)/2];

    qpts = [[-1, 1, 1,-1;
             -1,-1, 1, 1]/sqrt(3);
              1, 1, 1, 1];

    K = zeros(2*length(coor));
    f = zeros(2*length(coor),1);

    for c = conn
        cc = zeros(8,1);
        cc(1:2:end) = 2*c-1;
        cc(2:2:end) = 2*c;
        xe = coor(:,c);
        ke = zeros(8);
        for q = qpts
            dNdp = gradshape(q);
            J = xe * dNdp;
            dNdx = dNdp/J; 

            B = zeros(3,8);
            B(1,1:2:end) = dNdx(:,1);
            B(2,2:2:end) = dNdx(:,2);
            B(3,1:2:end) = dNdx(:,2);    
            B(3,2:2:end) = dNdx(:,1);

            ke = ke + B'*Ce*B*det(J)*q(3);
        end
        
        K(cc,cc) = K(cc,cc) + ke;
    end
    
    %traction force
    right_nodes = find(coor(1,:) == max(coor(1,:))); 
    for c = [right_nodes(1:end-1); right_nodes(2:end)]
        xe = coor(:,c);
        le = norm(xe(:,2)-xe(:,1));
        for q = [-1,1]/sqrt(3)
            N = 0.5*[1-q; 1+q];
            xq = xe*N;
            tbar = 200e6*xq(2);
            f(2*c-1) = f(2*c-1) + N*tbar*le/2;
        end
    end

    %boundary conditions
    left_nodes  = find(coor(1,:) == min(coor(1,:)));
%     size(left_nodes)
    for c = left_nodes
        K(2*c-1,:) = 0;
        K(:,2*c-1) = 0;
        K(2*c-1,2*c-1) = 1.0;
        f(2*c-1) = 0;
    end
    
    x = find(coor(1,:) == min(coor(1,:)));
    y = find(coor(2,:) == min(coor(2,:)));
    c = intersect(x,y);
    K(2*c,:) = 0;
    K(:,2*c) = 0;
    K(2*c,2*c) = 1.0;
    f(2*c) = 0;

    %solution
    d = K\f;
    
    figure(1)
    patch('vertices', coor', 'faces', conn', 'facecolor', 'interp', ...
        'facevertexcdata', d(1:2:end)); %displacement on x direction
    colorbar;
    figure(2)
    patch('vertices', coor', 'faces', conn', 'facecolor', 'interp', ...
        'facevertexcdata', d(2:2:end)); %displacement on y direction
    colorbar;
    
    %calculating sigma
    n = length(coor(1,:)); %number of nodes
    yI = zeros(n,3);
    AIJ = zeros(n);

    for c = conn
        cc = zeros(8,1);
        cc(1:2:end) = 2*c-1;
        cc(2:2:end) = 2*c;
        xe = coor(:,c);
        ye = zeros(4,3);
        aij = zeros(4,4);
        for q = qpts
            dNdp = gradshape(q);
            J = xe * dNdp;
            dNdx = dNdp/J;
            N = shape(q);
            B = zeros(3,8);
            B(1,1:2:end) = dNdx(:,1);
            B(2,2:2:end) = dNdx(:,2);
            B(3,1:2:end) = dNdx(:,2);
            B(3,2:2:end) = dNdx(:,1);
            
            sigma_h = Ce*B*d(cc)*det(J)*q(3);
            ye = ye + N*sigma_h';
            aij = aij + N*N'*det(J)*q(3);
        end
        yI(c,:) = yI(c,:) + ye;
        AIJ(c,c) = AIJ(c,c) + aij;
    end
    
    sigma = AIJ\yI;
    
    figure(3)
    patch('vertices', coor', 'faces', conn', 'facecolor', 'interp', ...
        'facevertexcdata', sigma(:,1));
    colorbar;

    
%     %right boundary plane
%     A = [];
%     B = sort(coor(2,right_nodes));
%     for i = 1:length(B)
%         for j = 1:n
%             if(coor(2,j) == B(i) && coor(1,j) >= 1.49)
%                 A(:,end+1) = sigma(j,1);
%             end
%         end
%     end
%     x = linspace(-0.5,0.5,50);
%     y = linspace(-1e8,1e8,50);
%     figure(2)
%     plot(B,A,'o-');
%     hold on
%     plot(x,y,'r');

    %A-A' plane
    A = [];
    center_node = find(abs(coor(1,:)) <= 1e-6);
    B = sort(coor(2,center_node));

    for i = 1:length(B)
        for j = 1:n
            if(coor(2,j) == B(i) && abs(coor(1,j)) <= 1e-6)
                A(:,end+1) = sigma(j,1);
            end
        end
    end
    figure(4)
    plot(B,A,'o-');
    
    %find out critical radius such that failure is equally happen at the
    %outside and the hole
    x = find(abs(coor(1,:)) <= 1e-5);
    y = find(abs(coor(2,:)-r) <= 1e-5);
    i = intersect(x,y);
    diff = sigma(i,1) - 1e8;
    
    %ductile failure
    
    
end