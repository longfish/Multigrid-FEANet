import numpy as np

import torch
import torch.nn.functional as F

from FEANet.mesh import MeshHandler
from FEANet.model import KNet, FNet

class MultigridBlock():
  '''
  Multigrid.
  '''
  def __init__(self):
    mesh = MeshHandler(outfile="Results/plate_mesh.vtk")
    self.model = KNet(mesh)