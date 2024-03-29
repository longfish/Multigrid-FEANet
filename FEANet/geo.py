import torch
import torch.nn as nn
import numpy as np

class Geometry():
    """Define square/L-shaped geometry for a given size"""
    def __init__(self, nnode_edge = 37, l_shape = False, l_cutout_size = None):
        if l_shape is False:
            self.square_geometry(nnode_edge)
        else:
            self.l_shaped_geometry(nnode_edge, l_cutout_size)

    def square_geometry(self, nnode_edge):
        """Defines a square geometry. Set the boundary-values to be zero"""

        # Define geometry, 1.0 inner points; 0.0 elsewhre
        self.geometry_idx = torch.ones(1, 1, nnode_edge ,nnode_edge )
        self.geometry_idx[0, 0,  0,  :] = torch.zeros(nnode_edge)
        self.geometry_idx[0, 0, -1,  :] = torch.zeros(nnode_edge)
        self.geometry_idx[0, 0,  :,  0] = torch.zeros(nnode_edge)
        self.geometry_idx[0, 0,  :, -1] = torch.zeros(nnode_edge)

        # Define boundary values
        self.boundary_value = torch.zeros_like(self.geometry_idx)

        # Set the boundary value to be 0
        self.boundary_value[0, 0,  0, :] = 0.0
        self.boundary_value[0, 0,  :,-1] = 0.0
        self.boundary_value[0, 0, -1, :] = 0.0
        self.boundary_value[0, 0, :, 0] = 0.0 
    
    def set_square_bc(self, bc_values):
        '''Input bc is a 2D array, only the locations at boundaries have values'''
        self.boundary_value[:, :, :, :] = bc_values


    def l_shaped_geometry(self, nnode_edge, l_cutout_size=None):
        """ Defines a L-shaped geometry of given size (think of creating the L-shape as cutting out a smaller square piece) """
        
        l_cutout_size = l_cutout_size or int(np.floor(nnode_edge / 2)) # l_cutout_size is by default size/2
        geometry_idx, boundary_value = self.square_geometry(nnode_edge)

        _, cutout_boundary_value = self.square_geometry(l_cutout_size)
        boundary_value[0, 0, :l_cutout_size, :l_cutout_size] = cutout_boundary_value
        boundary_value[0, 0, :l_cutout_size - 1, :l_cutout_size - 1] = torch.zeros(1)
        geometry_idx[0, 0, :l_cutout_size, :l_cutout_size] = torch.zeros(1)

        return geometry_idx, boundary_value