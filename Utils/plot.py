import numpy as np
import matplotlib.pyplot as plt

def plot_field(img,fname=None,limit=None):
    '''Plot the 2D pytorch tensor field'''
    if (len(img.shape) == 4):
        _, _, _, nnode_edge = img.shape
        img = img.view(nnode_edge, nnode_edge).numpy()
        
    fig = plt.figure()
    if(limit is None):
        im = plt.imshow(img)
    else:
        im = plt.imshow(img,vmin=limit[0],vmax=limit[1])
    
    ax = plt.gca()
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    plt.axis('off')
    plt.colorbar(im)
    if fname is not None:
        fig.savefig(r'./'+fname, bbox_inches='tight')
    plt.draw()
    plt.show()

def plot_pattern(mesh, key = 0):
    """Plot the distribution of nodes with pattern key"""
    pattern_center = mesh.points[np.where(mesh.global_pattern_center[key]==1)]
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.scatter(pattern_center[:,0],pattern_center[:,1],color = 'k',s=2)
    axes.axis('scaled')
    axes.set_xlim([-1,1])
    axes.set_ylim([-1,1])
    axes.set_title('Nodes distribution')