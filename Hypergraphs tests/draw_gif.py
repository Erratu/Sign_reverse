import networkx as nx
import itertools
import gif
import numpy as np
import imageio as iio
from os import listdir
import matplotlib.pyplot as plt


def draw_2d_simplicial_complex(simplices, pos=None, return_pos=False, ax = None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].
        
        Args
        ----
        simplices: list of lists of integers
            List of simplices to draw. Sub-simplices are not needed (only maximal).
            For example, the 2-simplex [1,2,3] will automatically generate the three
            1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
            When a higher order simplex is entered only its sub-simplices
            up to D=2 will be drawn.
        
        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.
           
        ax: matplotlib.pyplot.axes (default=None)
        
        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.
            
        References
        ----------    
        .. [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2019)
               "Simplicial Models of Social Contagion".
               Nature communications, 10(1), 2485.
    """

    
    #List of 0-simplices
    nodes =list(set(itertools.chain(*simplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))
    
    if ax is None: ax = plt.gca()
    ax.set_xlim([-4, 4])      
    ax.set_ylim([-4, 4])
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
       
    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_edges_from(edges)
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G)
        
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'black', zorder = 1, lw=0.7)
        ax.add_line(line);
    
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'black', facecolor = plt.cm.Blues(0.6),
                          zorder = 2, alpha=0.3, lw=0.5)
        ax.add_patch(tri);

    # Drawing the nodes 
    for i in nodes:
        (x, y) = pos[i]
        circ = plt.Circle([ x, y ], radius = 0.2, zorder = 3, lw=0.5,
                          edgecolor = 'Black', facecolor = u'#ff7f0e')
        ax.add_patch(circ);
        
def frame(complexT: list, t: int, pos: list, title: str, image_path: str = None, gif = False):
    '''
    

    Parameters
    ----------
    complexT : List
        List of simplicial complex along t
        
    t : int
        Time of vizualisation
        
    pos : list
        Position of each node. Take the followinf shape:
            [(x1,y1),(x2,y2),...,(xd,yd)]
    title : str
        Title for the figure
        
    image_path : str, optional
        Path for a background image. The default is None.

    Returns
    -------
    None.

    '''
    if image_path:
        img = plt.imread(image_path)
    plt.figure(figsize = (15,15))
    ax = plt.subplot(111)
    if image_path:
        ax.imshow(img, extent=[-4, 4, -4, 4])
    draw_2d_simplicial_complex([s[0] for s in complexT[t].get_simplices() if len(s[0])>2],
                                         pos = pos, ax = ax)
    plt.title(title)
    if not gif:
        plt.show()

def draw_gif(complexT,name,pos,duration,indic,k):
    gif.options.matplotlib["dpi"] = 300

    for i in range(round(len(complexT)/(k*30))):
        if complexT[i*60] is not None:
            frame(complexT,i*k*30,pos = pos,indic = indic,k = k)
    frames = np.stack([iio.imread("images/"+image) for image in listdir('images')], axis = 0)
    iio.mimwrite(name+'.gif', frames)
    #frames = [frame(complexT,i*30) for i in range(round(len(complexT)/30)) if complexT[i*30] is not None]
    #gif.save(frames, name+'.gif', duration=duration)