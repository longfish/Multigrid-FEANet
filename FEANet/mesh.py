import numpy as np
import meshio

class MeshCircInterface():
    """ Handle the square mesh with a circular region inside, quad elements
    0: background square; 1: circular region
    
    Parameters:
        size: length of the square plate
        nnode_edge: number of nodes along edges of the plate
    """

    def __init__(self, size = 2, prop = [1,20], nnode_edge = 65, shape = 0, outfile = None):
        self.size = size
        self.nnode_edge = nnode_edge
        self.points, self.cells, self.mesh = None, None, None
        self.global_pattern_center = {} # Dictionary to store global pattern maps, center nodes
        self.phase = np.zeros(((nnode_edge-1)*(nnode_edge-1),),dtype=int)
        self.pattern = np.zeros((nnode_edge*nnode_edge,4),dtype=int) # Pattern indices of each node, 
                                                                     # e.g., [1,0,1,0] means e1 and e3 are phase-1
        self.a = np.array(prop, dtype=np.float32) # Coefficients of linear Laplace equation
        # Reference pattern dictionary
        self.ref_pattern_dict = {0:[0,0,0,0],1:[1,1,1,1],2:[0,0,0,1],3:[0,0,1,0],
                                 4:[1,0,0,0],5:[0,1,0,0],6:[0,0,1,1],7:[1,1,0,0],
                                 8:[0,1,1,0],9:[1,0,0,1],10:[0,1,0,1],11:[1,0,1,0],
                                 12:[1,1,1,0],13:[1,1,0,1],14:[0,1,1,1],15:[1,0,1,1]}
        # Reference element stiffness matrix
        self.Ke = - 1./6.*np.array([[-4.,1.,2.,1.],
                                    [1.,-4.,1.,2.],
                                    [2.,1.,-4.,1.],
                                    [1.,2.,1.,-4.]], dtype=np.float32)
        self.kernel_dict = {} # Dictionary to store pytorch kernels
        self.generate_mesh()
        if (shape == 0):
            self.place_circle()
        elif(shape == 1):
            self.place_rect()

        self.identify_patterns()
        self.generate_global_pattern_map()
        self.generate_kernel()
        if outfile is not None:
            self.save_mesh(outfile)

    def generate_mesh(self):
        x = np.linspace(self.size/2,-self.size/2,self.nnode_edge, dtype=np.float32)
        y = np.linspace(-self.size/2,self.size/2,self.nnode_edge, dtype=np.float32)
        ms_x, ms_y = np.meshgrid(x,y)
        x = np.ravel(ms_x).reshape(-1,1)
        y = np.ravel(ms_y).reshape(-1,1)
        z = np.zeros_like(x, dtype=np.float32)
        self.points = np.concatenate((x,y,z),axis=1)
        n_element = (self.nnode_edge-1)*(self.nnode_edge-1)
        nodes = np.linspace(0,self.points.shape[0],self.points.shape[0],endpoint=False,dtype=int).reshape(self.nnode_edge,self.nnode_edge)
        self.cells = np.zeros((n_element,4),dtype=int)
        self.cells[:,0] = np.ravel(nodes[:self.nnode_edge-1,:self.nnode_edge-1])
        self.cells[:,1] = np.ravel(nodes[:self.nnode_edge-1,1:])
        self.cells[:,2] = np.ravel(nodes[1:,1:])
        self.cells[:,3] = np.ravel(nodes[1:,:self.nnode_edge-1])
        self.mesh = meshio.Mesh(self.points, [("quad",self.cells)])

    def place_circle(self, center=[0.2, 0.2], radius=0.5):
        for i in range(self.phase.shape[0]):
            element_centroid = np.mean(self.points[self.cells[i]],axis=0)
            r = (element_centroid[0]-center[0])**2+(element_centroid[1]-center[1])**2
            if(r < radius**2):
                self.phase[i] = 1
        self.mesh.cell_data['Phase'] = self.phase

    def place_rect(self, center=[0, 0], r=0.5):
        for i in range(self.phase.shape[0]):
            element_centroid = np.mean(self.points[self.cells[i]],axis=0)
            #r = (element_centroid[0]-center[0])**2+(element_centroid[1]-center[1])**2
            if(abs(element_centroid[0]-center[0]) < r and abs(element_centroid[1]-center[1]) < r):
                self.phase[i] = 1
        self.mesh.cell_data['Phase'] = self.phase

    def identify_patterns(self):
        for pid, pt in enumerate(self.points):
            elems = np.where(self.cells == pid)[0] # Find elements centered by the node
            if elems.shape[0] < 4:
                continue # Ignore the boundary nodes
            for eid in elems:
                if self.phase[eid] == 1:
                    el_centroid = np.mean(self.points[self.cells[eid]],axis=0)
                    if el_centroid[0]<pt[0] and el_centroid[1]<pt[1]:
                        self.pattern[pid,0] = 1 # Identified element-e1
                    if el_centroid[0]>pt[0] and el_centroid[1]<pt[1]:
                        self.pattern[pid,1] = 1 # Identified element-e2
                    if el_centroid[0]>pt[0] and el_centroid[1]>pt[1]:
                        self.pattern[pid,2] = 1 # Identified element-e3
                    if el_centroid[0]<pt[0] and el_centroid[1]>pt[1]:
                        self.pattern[pid,3] = 1 # Identified element-e4

    def generate_global_pattern_map(self):
        """Find nodes associated with each pattern: center nodes and surrounding nodes"""
        for pkey in self.ref_pattern_dict:
            self.global_pattern_center[pkey] = np.zeros((self.nnode_edge*self.nnode_edge,),dtype=int)
            for pid in range(self.points.shape[0]):
                if np.array_equal(self.ref_pattern_dict[pkey],self.pattern[pid]):
                    self.global_pattern_center[pkey][pid] = 1

    def generate_kernel(self):
        for pkey in self.ref_pattern_dict:
            kernel = np.zeros((3,3), dtype=np.float32)
            pattern = self.ref_pattern_dict[pkey] # The pattern has the form of [0,1,0,0]
            kernel[0,0] = self.a[pattern[3]]*self.Ke[1,3] # Ke4_24
            kernel[0,1] = self.a[pattern[3]]*self.Ke[1,2] + self.a[pattern[2]]*self.Ke[0,3] # Ke4_23+Ke3_14
            kernel[0,2] = self.a[pattern[2]]*self.Ke[0,2] # Ke3_13
            kernel[1,0] = self.a[pattern[0]]*self.Ke[2,3] + self.a[pattern[3]]*self.Ke[1,0] # Ke1_34+Ke4_21
            kernel[1,1] = self.a[pattern[2]]*self.Ke[0,0] + self.a[pattern[3]]*self.Ke[1,1] + \
                          self.a[pattern[0]]*self.Ke[2,2] + self.a[pattern[1]]*self.Ke[3,3] # Ke3_11+Ke4_22+Ke1_33+Ke2_44
            kernel[1,2] = self.a[pattern[1]]*self.Ke[3,2] + self.a[pattern[2]]*self.Ke[0,1] # Ke2_43+Ke3_12
            kernel[2,0] = self.a[pattern[0]]*self.Ke[2,0] # Ke1_31
            kernel[2,1] = self.a[pattern[0]]*self.Ke[2,1] + self.a[pattern[1]]*self.Ke[3,0] # Ke1_32+Ke2_41
            kernel[2,2] = self.a[pattern[1]]*self.Ke[3,1] # Ke2_42
            self.kernel_dict[pkey] = kernel

    def save_mesh(self,outfile = 'plate_mesh.vtk'):
        self.mesh.write(outfile)

class MeshSquare():
    """ Handle the square mesh with quad elements
    0: background square; 1: circular region
    
    Parameters:
        size: length of the square plate
        nnode_edge: number of nodes along edges of the plate
    """

    def __init__(self, size = 2, prop = [1,20], nnode_edge = 65, shape = 0, outfile = None):
        self.size = size
        self.nnode_edge = nnode_edge
        self.points, self.cells, self.mesh = None, None, None
        self.global_pattern_center = {} # Dictionary to store global pattern maps, center nodes
        self.phase = np.zeros(((nnode_edge-1)*(nnode_edge-1),),dtype=int)
        self.pattern = np.zeros((nnode_edge*nnode_edge,4),dtype=int) # Pattern indices of each node, 
                                                                     # e.g., [1,0,1,0] means e1 and e3 are phase-1
        self.a = np.array(prop, dtype=np.float32) # Coefficients of linear Laplace equation
        # Reference pattern dictionary
        self.ref_pattern_dict = {0:[0,0,0,0],1:[1,1,1,1],2:[0,0,0,1],3:[0,0,1,0],
                                 4:[1,0,0,0],5:[0,1,0,0],6:[0,0,1,1],7:[1,1,0,0],
                                 8:[0,1,1,0],9:[1,0,0,1],10:[0,1,0,1],11:[1,0,1,0],
                                 12:[1,1,1,0],13:[1,1,0,1],14:[0,1,1,1],15:[1,0,1,1]}
        # Reference element stiffness matrix
        self.Ke = - 1./6.*np.array([[-4.,1.,2.,1.],
                                    [1.,-4.,1.,2.],
                                    [2.,1.,-4.,1.],
                                    [1.,2.,1.,-4.]], dtype=np.float32)
        self.kernel_dict = {} # Dictionary to store pytorch kernels
        self.generate_mesh()
        if (shape == 0):
            self.place_circle()
        elif(shape == 1):
            self.place_rect()

        self.identify_patterns()
        self.generate_global_pattern_map()
        self.generate_kernel()
        if outfile is not None:
            self.save_mesh(outfile)

    def generate_mesh(self):
        x = np.linspace(self.size/2,-self.size/2,self.nnode_edge, dtype=np.float32)
        y = np.linspace(-self.size/2,self.size/2,self.nnode_edge, dtype=np.float32)
        ms_x, ms_y = np.meshgrid(x,y)
        x = np.ravel(ms_x).reshape(-1,1)
        y = np.ravel(ms_y).reshape(-1,1)
        z = np.zeros_like(x, dtype=np.float32)
        self.points = np.concatenate((x,y,z),axis=1)
        n_element = (self.nnode_edge-1)*(self.nnode_edge-1)
        nodes = np.linspace(0,self.points.shape[0],self.points.shape[0],endpoint=False,dtype=int).reshape(self.nnode_edge,self.nnode_edge)
        self.cells = np.zeros((n_element,4),dtype=int)
        self.cells[:,0] = np.ravel(nodes[:self.nnode_edge-1,:self.nnode_edge-1])
        self.cells[:,1] = np.ravel(nodes[:self.nnode_edge-1,1:])
        self.cells[:,2] = np.ravel(nodes[1:,1:])
        self.cells[:,3] = np.ravel(nodes[1:,:self.nnode_edge-1])
        self.mesh = meshio.Mesh(self.points, [("quad",self.cells)])

    def place_circle(self, center=[0.2, 0.2], radius=0.5):
        for i in range(self.phase.shape[0]):
            element_centroid = np.mean(self.points[self.cells[i]],axis=0)
            r = (element_centroid[0]-center[0])**2+(element_centroid[1]-center[1])**2
            if(r < radius**2):
                self.phase[i] = 1
        self.mesh.cell_data['Phase'] = self.phase

    def place_rect(self, center=[0, 0], r=0.5):
        for i in range(self.phase.shape[0]):
            element_centroid = np.mean(self.points[self.cells[i]],axis=0)
            #r = (element_centroid[0]-center[0])**2+(element_centroid[1]-center[1])**2
            if(abs(element_centroid[0]-center[0]) < r and abs(element_centroid[1]-center[1]) < r):
                self.phase[i] = 1
        self.mesh.cell_data['Phase'] = self.phase

    def identify_patterns(self):
        for pid, pt in enumerate(self.points):
            elems = np.where(self.cells == pid)[0] # Find elements centered by the node
            if elems.shape[0] < 4:
                continue # Ignore the boundary nodes
            for eid in elems:
                if self.phase[eid] == 1:
                    el_centroid = np.mean(self.points[self.cells[eid]],axis=0)
                    if el_centroid[0]<pt[0] and el_centroid[1]<pt[1]:
                        self.pattern[pid,0] = 1 # Identified element-e1
                    if el_centroid[0]>pt[0] and el_centroid[1]<pt[1]:
                        self.pattern[pid,1] = 1 # Identified element-e2
                    if el_centroid[0]>pt[0] and el_centroid[1]>pt[1]:
                        self.pattern[pid,2] = 1 # Identified element-e3
                    if el_centroid[0]<pt[0] and el_centroid[1]>pt[1]:
                        self.pattern[pid,3] = 1 # Identified element-e4

    def generate_global_pattern_map(self):
        """Find nodes associated with each pattern: center nodes and surrounding nodes"""
        for pkey in self.ref_pattern_dict:
            self.global_pattern_center[pkey] = np.zeros((self.nnode_edge*self.nnode_edge,),dtype=int)
            for pid in range(self.points.shape[0]):
                if np.array_equal(self.ref_pattern_dict[pkey],self.pattern[pid]):
                    self.global_pattern_center[pkey][pid] = 1

    def generate_kernel(self):
        for pkey in self.ref_pattern_dict:
            kernel = np.zeros((3,3), dtype=np.float32)
            pattern = self.ref_pattern_dict[pkey] # The pattern has the form of [0,1,0,0]
            kernel[0,0] = self.a[pattern[3]]*self.Ke[1,3] # Ke4_24
            kernel[0,1] = self.a[pattern[3]]*self.Ke[1,2] + self.a[pattern[2]]*self.Ke[0,3] # Ke4_23+Ke3_14
            kernel[0,2] = self.a[pattern[2]]*self.Ke[0,2] # Ke3_13
            kernel[1,0] = self.a[pattern[0]]*self.Ke[2,3] + self.a[pattern[3]]*self.Ke[1,0] # Ke1_34+Ke4_21
            kernel[1,1] = self.a[pattern[2]]*self.Ke[0,0] + self.a[pattern[3]]*self.Ke[1,1] + \
                          self.a[pattern[0]]*self.Ke[2,2] + self.a[pattern[1]]*self.Ke[3,3] # Ke3_11+Ke4_22+Ke1_33+Ke2_44
            kernel[1,2] = self.a[pattern[1]]*self.Ke[3,2] + self.a[pattern[2]]*self.Ke[0,1] # Ke2_43+Ke3_12
            kernel[2,0] = self.a[pattern[0]]*self.Ke[2,0] # Ke1_31
            kernel[2,1] = self.a[pattern[0]]*self.Ke[2,1] + self.a[pattern[1]]*self.Ke[3,0] # Ke1_32+Ke2_41
            kernel[2,2] = self.a[pattern[1]]*self.Ke[3,1] # Ke2_42
            self.kernel_dict[pkey] = kernel

    def save_mesh(self,outfile = 'plate_mesh.vtk'):
        self.mesh.write(outfile)