import torch
import numpy as np

class JacobiBlock():
    """ Define all the methods necessary for a CNN-based Jacobi iteration 
        
        Knet: neural network model for stiffness terms
        mesh: an object that define the mesh
        initial_u : tensor-like, shape = [*, *, n, n]
            Initial values of the variable.
        geometry_idx : tensor-like, shape = [*, *, n, n]
            Matrix describing the domain: 1.0 for inner points 0.0 elsewhere.
        boundary_value: tensor-like, shape = [*, *, n, n]
            Matrix describing the domain: desired values for boundary points 0.0 elsewhere.
        n_iter: number of Jacobi iterations
    """
    def __init__(self, Knet, mesh, omega, geometry_idx, boundary_value):
        self.nnode_edge = geometry_idx.shape[2]
        self.geometry_idx = geometry_idx
        self.boundary_value = boundary_value
        self.omega = omega
        self.mesh = mesh
        self.d_mat = torch.zeros((1, 1, self.nnode_edge, self.nnode_edge)) # Diagonal matrix for Jacobi iteration
        self.compute_diagonal_matrix()
        self.Knet = Knet # Initialize the stiffness network, given mesh

    def reset_boundary(self, u):
        """ Reset values at the boundary of the domain """
        return u * self.geometry_idx + self.boundary_value

    def compute_diagonal_matrix(self):
        """ Comopute diagonal matrix for Jacobi iteration """
        for pkey in self.mesh.kernel_dict:
            K_weights = torch.from_numpy(self.mesh.kernel_dict[pkey]) ## 3x3 size kernel
            global_pattern = torch.from_numpy(self.mesh.global_pattern_center[pkey]).reshape(self.nnode_edge, self.nnode_edge)
            self.d_mat[0,0,:,:] += global_pattern*K_weights[1,1] 

    def jacobi_iteration_step(self, u, forcing_term):
        """ Jacobi method iteration step defined as a convolution:
        u_new = omega/d_mat*residual + u, where residual = f - K*u (* is convolution operator here)
        note that the forcing_term should be already convoluted, i.e., forcing_term = fnet(f), when source term is f
        """
        residual = forcing_term-self.Knet(u)
        u_new = self.omega/self.d_mat*residual + u
        return self.reset_boundary(u_new)

    def jacobi_convolution(self, initial_u, forcing_term, n_iter = 1000):
        """ Compute jacobi method solution by convolution. 

            Return: 
                u """

        u = self.reset_boundary(initial_u)
        #error = np.zeros((n_iter,))
        for i in range(n_iter):
            #u_prev = u
            u = self.jacobi_iteration_step(u, forcing_term)
            #error[i] = torch.sqrt(torch.sum((u - u_prev) ** 2)).item() / torch.sqrt(torch.sum((u) ** 2)).item()

        return u #, error