import numpy as np
import networkx as nx
from mpi4py import MPI
import matplotlib.pyplot as plt


# ---------------------Funktion: Den kuerzesten Pfad von unten nach oben mit 'Dijkstra Algorithmus' suchen--------------
def calculate_shortest_path(graph, target_face_list, source_nodes):
    '''
    Calculation of the shortest path with the Dijkstra algorithm
    Function written by Shihai Liu (Forschungspraktikum)
    '''
    try:
        shortest_length, shortest_path = nx.multi_source_dijkstra(graph, target_face_list, source_nodes)
        return shortest_length, shortest_path, source_nodes
    # Situation, dass der Pfad vom Startpunkt zum Endpunkt nicht erreichbar ist, None zurückgeben.
    except:
        return None, None, None
    

# -----------------------------------Funktion: Den kaerzesten Pfad visualisieren----------------------------------------
def visualization_of_shortest_path(dict):
    # 3-Achse-Koordinaten erstellen
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Startpunkte und Pfade zeichnen
    for key, points in dict.items():  # SchlÃ¼ssel und Werte in dictionary durchlaufen
        # Koordinate x, y, z von Werte einlesen
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        z_vals = [point[2] for point in points]

        # Pfade zeichnen
        ax.plot(x_vals, y_vals, z_vals, label=key)

        # Koordinate x, y, z von SchlÃ¼ssel einlesen
        x_vals = key[0]
        y_vals = key[1]
        z_vals = key[2]

        # Startpunkte zeichnen
        ax.scatter(x_vals, y_vals, z_vals)

    # Etiketten und Titel setzen
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Paths')


# def skeletonize2D(_image: np.ndarray, _direction_flow: int, _starting_depth: int =0, 
#                   _plot_skel: bool = False, _fname: str = None, _DIRname: str = None):
def skeletonize2D(_image: np.ndarray):
    '''
    Function to skeletonize the voxel-based image according to SSPSM


    _image :            SEGMENTED voxel-based image (either 0 or 1 for each phase)
    _direction_flow :   tortuosity in a specific direction
    _starting_depth :   Starting depth of finding starting points (default: 0)
    _plot_skel :        Plot and save skeleton figure (default: False)
    _DIRname:           Name of Directory
    _fname:             Filename


    medial_axis_image :             voxel-based image with highlighted skeleton
    nrows :                         number of rows
    ncolumns :                      number of columns
    ndepth :                        number of "layers"
    cum_number_medial_each_slide :  number of skeleton voxels for each slide
    skeleton_volume_axis_index :    indices of skeleton voxels
    '''
        
    # dimensions of image
    nrows = _image.shape[0]       # number of rows (x)
    ncolumns = _image.shape[1]    # number of columns (y)
    ndepth = _image.shape[2]      # number of slices in z (z)
    
    # create empty lists
    skeleton_volume_axis_index = []                   
    number_medial_each_slide = []
    medial_axis_location_all = []
    
    # medial axis image generation 
    # imagse_orig = np.copy(image)
    medial_axis_image = _image
    
    # skeletonization
    for depth in range(ndepth):
        
        binary_slice = _image[:, :, depth]                          # Extract the 2D slice
        binary_slice = binary_slice.astype(bool).copy(order='C')    # Ensure the binary slice is in the correct format (dtype=bool) and C-contiguous
        medial_skeleton_slice = skeletonize(1-binary_slice)         # Perform skeletonization on the binary slice
        
        
        medial_axis_location = np.where(medial_skeleton_slice==True)# find all values in array ==1 (list of index; looks at first coloumn, then in second coloumn etc.)
        medial_axis_indices_slice = np.sort(medial_axis_location[0] + nrows*medial_axis_location[1]) # local index; looks at first coloumn, then in second coloumn etc.
        
        medial_axis_location_all.append(medial_axis_location)
        number_medial_each_slide.append(len(medial_axis_indices_slice))
        skeleton_volume_axis_index.extend(medial_axis_indices_slice+nrows*ncolumns*depth)            # Append the skeletonized slice to the result volume
         
        for coor in range(len(medial_axis_location[0])):
            x_coor = medial_axis_location[0][coor]
            y_coor = medial_axis_location[1][coor]
            medial_axis_image[x_coor, y_coor, depth] = 2
            
    # Convert the list of skeletonized slices back to a 3D array
    skeleton_volume_axis_index = np.array(skeleton_volume_axis_index)
    cum_number_medial_each_slide = np.cumsum(number_medial_each_slide)                  # number of skeleton voxels per slide summed up
    number_medial_each_slide = np.insert(number_medial_each_slide, 0, [0, 0])           # adds 1,1 before vector
    cum_number_medial_each_slide = np.insert(cum_number_medial_each_slide, 0, [0, 0])   # adds 1,1 before vector
    
    # if _plot_skel:
        # plot of the skeleton of the first slide
        # skeleton_first_slide = medial_axis_image[:, :, 0]
        # plot_fig(skeleton_first_slide, 'Skeleton_first_slide_'+_fname, _DIRname)
        
    return medial_axis_image


# -----------------------------------Funktion: Statistische Werte der Tortuosität----------------------------------------  
def plot_tort_distribution(tort_values: np.ndarray,
                           path_to_file: str, _filename: str):
    try:
        tortu_min = np.min(tort_values)
        tortu_max = np.max(tort_values)
        
        segments = np.linspace(tortu_min,tortu_max,20)
    
        segment_numbers = []
        for i in range(len(segments)-1):
            number_of_values = np.count_nonzero((tort_values >= segments[i]) & (tort_values <= segments[i+1]))
            segment_numbers.append(number_of_values)
	    
        probability = np.array(segment_numbers)/len(tort_values)
	    
        segments_avg = (segments[:-1]+segments[1:])/2
	    
        segments = np.round(segments,4)
        plt.figure(figsize=(10,8), dpi=200)
        plt.bar(segments_avg, probability*100, width=0.01)
        plt.xticks(segments, [f"{segments[0]}", "",  "", "", f"{segments[4]}", "", "", "", "", f"{segments[9]}", "", "", "", "", f"{segments[14]}", "", "", "", "", f"{segments[19]}"])
        plt.xlabel(r'Tortuosity $\tau$')
        plt.ylabel('Probability in $\%$')   
        plt.savefig(path_to_file+_filename+".pdf")

    except:
        print("No tortuosity values.")


# ------------------------Funktion: 3D-Matrix auf ungerichteten Graphen nach Konnektivitaet transformieren--------------
def array_to_graph(array: np.array, phase: int, connectivity: int, lx: float, ly: float, lz: float, start, end):
    '''
    Function to generate a graph that is needed for the Dijkstra Algorithm
    Function from Shihai Liu (Forschungspraktikum)

    Parameters
    ----------
    array : Array of the image
    phase : Phase that should be considered (e.g. Pore=0,)
    connectivity : connectivity that should be considered between the voxels (3D: 6, 18, 28)
    lxy : length of voxel in xy direction (resolution)
    lz : length of voxel in z direction (resolution)
    start : start layer for each rank
    end : end layer for each rank -> connects voxel from start to end

    Returns
    -------
    graph : Graph - for Dijksta algorithm

    '''

    graph = nx.Graph()  # einen leeren Graphen erstellen

    # 3D-Matrix durchlaufen, um die gesuchte Phase in Graphen als Knoten hinzufuegen
    for z, layer in enumerate(array):
        for y, row in enumerate(layer):
            for x, pixel in enumerate(row):
                if pixel == phase:
                    graph.add_node((x, y, start+z))  
                    # if z == 0:
                    #     graph.add_node((x, y, start))    
                    # else:graph.add_node((x, y, end+z))   # + z?
    # Knoten verbinden
    # 6 Konnektivitaet
    if connectivity == 6:
        # Unterschied zwischen einem Punkt und den nach 6-konnektivitaet verbundenen Nachbarpunkten
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        for node in graph.nodes:
            x, y, z = node
            neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in directions]
            for neighbor in neighbors:
                if neighbor in graph.nodes:
                    if node[0] != neighbor[0]:
                        # Abstand zwischen benachbarten Punkten in x Richtung
                        graph.add_edge(node, neighbor, weight=lx)
                    elif node[1] != neighbor[1]:
                        # Abstand zwischen benachbarten Punkten in y Richtung
                        graph.add_edge(node, neighbor, weight=ly)
                    elif node[2] != neighbor[2]:
                        # Abstand zwischen benachbarten Punkten in z Richtung
                        graph.add_edge(node, neighbor, weight=lz)
        return (graph)

    # 18 Konnektivitaet
    elif connectivity == 18:
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),

                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                      (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]

        for node in graph.nodes:
            x, y, z = node
            neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in directions]
            for neighbor in neighbors:
                if neighbor in graph.nodes:
                    if node[2] == neighbor[2]:
                        if node[0] != neighbor[0] and node[1] == neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die in der gleichen Schicht und
                            # vorne, hinten, links, rechts aufeinander liegen
                            graph.add_edge(node, neighbor, weight=lx)
                        elif node[0] == neighbor[0] and node[1] != neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die in der gleichen Schicht und
                            # vorne, hinten, links, rechts aufeinander liegen
                            graph.add_edge(node, neighbor, weight=ly)
                        else:
                            # Abstand zwischen benachbarten Punkten, die in der gleichen Schicht und
                            # diagonal aufeinander liegen
                            graph.add_edge(node, neighbor, weight=np.sqrt(lx ** 2 + ly ** 2))
                    elif node[2] != neighbor[2]:
                        if node[0] == neighbor[0] and node[1] == neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die nicht in der gleichen Schicht und
                            # oben und unten aufeinander liegen
                            graph.add_edge(node, neighbor, weight=lz)
                        elif node[0] != neighbor[0] and node[1] == neighbor[1]:
                            graph.add_edge(node, neighbor, weight=np.sqrt(lx ** 2 + lz ** 2))
                        elif node[0] == neighbor[0] and node[1] != neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die nicht in der gleichen Schicht und
                            # diagonal aufeinander liegen
                            graph.add_edge(node, neighbor, weight=np.sqrt(ly ** 2 + lz ** 2))
        return (graph)

    # 26 Konnektivitaet
    elif connectivity == 26:
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),

                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                      (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),

                      (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
                      (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)]

        for node in graph.nodes:
            x, y, z = node
            neighbors = [(x + dx, y + dy, z + dz) for dx, dy, dz in directions]  # Nachbarn von einer Pore zu finden
            for neighbor in neighbors:
                if neighbor in graph.nodes:
                    if node[2] == neighbor[2]:
                        if node[0] != neighbor[0] and node[1] == neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die in der gleichen Schicht und
                            # vorne, hinten, links, rechts aufeinander liegen
                            graph.add_edge(node, neighbor, weight=lx)
                        elif node[0] == neighbor[0] and node[1] != neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die in der gleichen Schicht und
                            # vorne, hinten, links, rechts aufeinander liegen
                            graph.add_edge(node, neighbor, weight=ly)
                        else:
                            # Abstand zwischen benachbarten Punkten, die in der gleichen Schicht und
                            # diagonal aufeinander liegen
                            graph.add_edge(node, neighbor, weight=np.sqrt(lx ** 2 + ly ** 2))
                    if node[2] != neighbor[2]:
                        if node[0] == neighbor[0] and node[1] == neighbor[1]:
                            graph.add_edge(node, neighbor, weight=lz)

                        elif node[0] != neighbor[0] and node[1] == neighbor[1]:
                            graph.add_edge(node, neighbor, weight=np.sqrt(lx ** 2 + lz ** 2))

                        elif node[0] == neighbor[0] and node[1] != neighbor[1]:
                            graph.add_edge(node, neighbor, weight=np.sqrt(ly ** 2 + lz ** 2))

                        else: # node[0] != neighbor[0] and node[1] != neighbor[1]:
                            # Abstand zwischen benachbarten Punkten, die in x, y und z Richtung anders sind
                            graph.add_edge(node, neighbor, weight=np.sqrt(lx ** 2 + ly ** 2 + lz ** 2))
        return (graph)

def create_graph_global(_array: np.ndarray, Vox_size_x: float, Vox_size_y: float, Vox_size_z: float,
                        celltags: list = [10, 11, 12], connectivity = 6, 
                        _save_graphs: bool = False, _fname=''):
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    Pore = celltags[0]
    Ni = celltags[1]
    YSZ = celltags[2]
    
    # 3D-Matrix nach Hoehe in jedem Prozess zerlegen
    hight = _array.shape[2]

    if rank == 0: 
        start = 0
    if rank !=0: 
        start = rank * (hight // size)-1
    end = (rank + 1) * (hight// size) if rank != size - 1 else hight

    # ungerichteten lokalen Graph nach gesuchter Pase und Konnektivitaet mit MPI bilden
    # ungerichteten lokalen Graph in Form von List als globaler Graph zusammenfassen
    local_graph_Pore = array_to_graph(_array[start:end], Pore, connectivity, Vox_size_x, Vox_size_y, Vox_size_z, start, end)
    global_graph_Pore = comm.gather(local_graph_Pore, root=0)

    local_graph_YSZ = array_to_graph(_array[start:end], YSZ, connectivity, Vox_size_x, Vox_size_y, Vox_size_z, start, end)
    global_graph_YSZ = comm.gather(local_graph_YSZ, root=0)

    local_graph_Ni = array_to_graph(_array[start:end], Ni, connectivity, Vox_size_x, Vox_size_y, Vox_size_z, start, end)
    global_graph_Ni = comm.gather(local_graph_Ni, root=0)

    # einen leeren Graphen erstellen
    final_graph_Pore = nx.Graph()
    final_graph_YSZ = nx.Graph()
    final_graph_Ni = nx.Graph()

    # Hautprozess
    if rank == 0:
        # Knoten und Raender von List des globalen Graphen zu endlichen Graphen uebertragen
        for g in global_graph_Pore:
            final_graph_Pore.add_nodes_from(g.nodes)
            final_graph_Pore.add_edges_from(g.edges(data=True))

        for g in global_graph_YSZ:
            final_graph_YSZ.add_nodes_from(g.nodes)
            final_graph_YSZ.add_edges_from(g.edges(data=True))

        for g in global_graph_Ni:
            final_graph_Ni.add_nodes_from(g.nodes)
            final_graph_Ni.add_edges_from(g.edges(data=True))

        # Graphen speichern
        if _save_graphs:
            file_name_Pore = f'{_fname}_graph_pore.pickle'
            file_name_YSZ = f'{_fname}_graph_YSZ.pickle'
            file_name_Ni = f'{_fname}_graph_Ni.pickle'
    
            with open(file_name_Pore, 'wb') as f: pickle.dump(final_graph_Pore, f)
            with open(file_name_YSZ, 'wb') as f: pickle.dump(final_graph_YSZ, f)
            with open(file_name_Ni, 'wb') as f: pickle.dump(final_graph_Ni, f)
    
            # Anzahl der Knoten, Raender und Matrixdimension ueberpruefen
            print("Anzahl von Knoten bei Pore:", final_graph_Pore.number_of_nodes())
            print("Anzahl von Raendern bei Pore:", final_graph_Pore.number_of_edges())
    
            print("Anzahl von Knoten bei YSZ:", final_graph_YSZ.number_of_nodes())
            print("Anzahl von Raendern bei YSZ:", final_graph_YSZ.number_of_edges())
    
            print("Anzahl von Knoten bei Ni:", final_graph_Ni.number_of_nodes())
            print("Anzahl von Raendern bei Ni:", final_graph_Ni.number_of_edges())
    
            total_nodes=final_graph_Pore.number_of_nodes()+final_graph_YSZ.number_of_nodes()+final_graph_Ni.number_of_nodes()
    
            # Anzahl der Knoten, Raender und Matrixdimension ueberpruefen - funktioniert nur bei DSPSM
            print('Volumenanteil von Pore (nur bei DSPSM sinnvoll!):', final_graph_Pore.number_of_nodes()/total_nodes)
            print('Volumenanteil von Pore (nur bei DSPSM sinnvoll!):', final_graph_YSZ.number_of_nodes()/total_nodes)
            print('Volumenanteil von Pore (nur bei DSPSM sinnvoll!):', final_graph_Ni.number_of_nodes()/total_nodes)

    final_graph_Pore = comm.bcast(final_graph_Pore, root=0)
    final_graph_Ni   = comm.bcast(final_graph_Ni,   root=0)
    final_graph_YSZ  = comm.bcast(final_graph_YSZ,  root=0)
            
    return final_graph_Pore, final_graph_Ni, final_graph_YSZ

def DSPSM(_array, _Size_of_Voxel_x: int, _Size_of_Voxel_y: int, _Size_of_Voxel_z: int, _connectivity: int , dir: int, _celltags = [10, 12, 11],
          _fname = '', _save_graphs = False, _save_paths = False, path=''):
    '''
    Direct Shortest Path Searching Method
    Implemented by Liu Shihai (Forschungspraktikum)
    
    '''
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  

    # number of voxels in each direction
    Nr_of_Voxel_x = _array.shape[0]
    Nr_of_Voxel_y = _array.shape[1]
    Nr_of_Voxel_z = _array.shape[2]    

    # create the global graphs
    ##########################
    graph_Pore, graph_Ni, graph_YSZ = create_graph_global(_array, Vox_size_x = _Size_of_Voxel_x, Vox_size_y = _Size_of_Voxel_y, 
                                                          Vox_size_z = _Size_of_Voxel_z, celltags = _celltags,
                                                          connectivity = _connectivity, _save_graphs = _save_graphs, _fname=path+_fname)


    # calculate the tortuosity for each graph / each phase
    ######################################################
        
    # list for tortuosities
    tort_list = [] # at first: tau_pore, tau_Ni, tau_YSZ
    
    for nr_i, graph_i in enumerate([graph_Pore, graph_Ni, graph_YSZ]):
        # Knoten in Quellflaeche und Zielflaeche aus Graphen finden
        array_total_nodes_each_layer = Nr_of_Voxel_x * Nr_of_Voxel_y
        source_nodes_list = [node for node in list(graph_i.nodes())[:array_total_nodes_each_layer] if
                             node[2] == 0]                  # Quellflaeche
        target_face_list = [node for node in list(graph_i.nodes())[-array_total_nodes_each_layer:] if
                             node[2] == Nr_of_Voxel_z - 1]  # Zielflaeche
    
        # Funktion calculate_shortest_path mit MPI auf verschiedenen Noten/CPUs aufrufen
        start = rank * (len(source_nodes_list) // size)
        end = (rank + 1) * (len(source_nodes_list) // size) if rank != size - 1 else len(source_nodes_list)
        local_results = [calculate_shortest_path(graph_i, target_face_list, source_nodes_list[i]) for i in range(start, end)]
        results = comm.gather(local_results, root=0)
    
        # Hautprozess
        if rank == 0:
            path_dict = {}
            average_length = 0
            number_of_path = 0
            shortest_lengths = []
    
            # Tortuositaet, Anzahl der kuerzesten Pfaden rechnen, die kuerzeste Pfade im dictionary speichern
            for result in results:
                for shortest_length, shortest_path, source_node in result:
                    if shortest_length is not None:
                        path_dict[source_node] = shortest_path
                        average_length += shortest_length
                        shortest_lengths.append(shortest_length/(_Size_of_Voxel_z * (Nr_of_Voxel_z-1)))
                        number_of_path += 1
            
            try:
                average_length /= number_of_path
            except:
                print(f"Not even one connected path for phase {nr_i}.")
                average_length = 0

            tortuosity_mean = average_length / (_Size_of_Voxel_z * (Nr_of_Voxel_z-1))
            tort_list.append(tortuosity_mean)
            
            if number_of_path > 0:
                tortuosity_std = np.std(np.array(shortest_lengths))                           # standard derivation of tortuosity path
                tortuosity_median = np.median(np.array(shortest_lengths))                     # median value of tortuosity 
            else:
                tortuosity_std = 0
                tortuosity_median = 0
    
            # Ergebnisse in einem Text speichern
            if _save_graphs:
                with open(f'{path}Tortuosity_{_fname}_results_DSPSM_Direction{dir}.txt', 'a') as result_Text:
                    result_Text.write( '___________________________________\n')
                    result_Text.write(f'___________Phase: {nr_i}___________\n')
                    result_Text.write(f'Konnektivitaet:{_connectivity}\n')
                    result_Text.write(f'Richtung:{dir}\n')
                    result_Text.write(f'Gemittelte Tortuositaet: {tortuosity_mean}\n')
                    result_Text.write(f'Standardabweichung Tortuositaet: {tortuosity_std}\n')
                    result_Text.write(f'Median Tortuositaet: {tortuosity_median}\n\n')
                    result_Text.write(f'Länge einzelner Pfade: {shortest_lengths} \n \n')
                    result_Text.write(f'Anzahl der verbundenen Pfade: {number_of_path}\n\n')
                    result_Text.write('Dictionary:\n')
                    for key, value in path_dict.items():
                        result_Text.write(f'{key}:{value}\n')
    
            # Pfade visualisieren und Distribution der Pfade
            if _save_paths and number_of_path > 0:
                visualization_of_shortest_path(path_dict)
                plt.savefig(f'{path}ShortestPath_{_fname}_DSPSM_{nr_i}_Direction{dir}.pdf')
                plot_tort_distribution(np.array(shortest_lengths), path, f"TortDistribution{_fname}_Phase{nr_i}_Direction{dir}")
                    
        
    return tort_list

def SSPSM(_array, _Size_of_Voxel_x, _Size_of_Voxel_y,_Size_of_Voxel_z, _connectivity, dir: int, _celltags = [10, 12, 11],
          _fname = '', _save_graphs = False, _save_paths = False, path=''):
    '''
    Skeleton Shortest Path Searching Method
    
    '''
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # number of voxels in each direction
    Nr_of_Voxel_x = _array.shape[0]
    Nr_of_Voxel_y = _array.shape[1]
    Nr_of_Voxel_z = _array.shape[2]
    
    # skeletonization of array
    ##########################
    image_pore = _array == _celltags[0]
    image_Ni   = _array == _celltags[1]
    image_YSZ  = _array == _celltags[2]
    
    # skeleton images for pores
    pore_skel = skeletonize2D((1-image_pore))
    Ni_skel   = skeletonize2D((1-image_Ni))
    YSZ_skel  = skeletonize2D((1-image_YSZ))
    
    # alle Skelette der 3 Phasen wieder in einer Matrix speichern
    array_skeleton = np.full((Nr_of_Voxel_z, Nr_of_Voxel_y, Nr_of_Voxel_x), 500, dtype=int)
    array_skeleton[pore_skel==2] = _celltags[0]
    array_skeleton[YSZ_skel ==2] = _celltags[2]
    array_skeleton[Ni_skel  ==2] = _celltags[1]
    
    
    # create the global graphs
    ##########################
    graph_Pore, graph_Ni, graph_YSZ = create_graph_global(array_skeleton, Vox_size_x = _Size_of_Voxel_x, 
                                Vox_size_y = _Size_of_Voxel_y, Vox_size_z = _Size_of_Voxel_z, celltags = _celltags,
                                connectivity = _connectivity, _save_graphs = _save_graphs, _fname=_fname)


    # calculate the tortuosity for each graph / each phase
    ######################################################
    
    # list for tortuosities
    tort_list = [] # at first: tau_pore, tau_Ni, tau_YSZ
    
    for nr_i, graph_i in enumerate([graph_Pore, graph_Ni, graph_YSZ]):
        # Knoten in Quellflaeche und Zielflaeche aus Graphen finden
        array_total_nodes_each_layer = Nr_of_Voxel_x * Nr_of_Voxel_y
        source_nodes_list = [node for node in list(graph_i.nodes())[:array_total_nodes_each_layer] if
                             node[2] == 0]                  # Quellflaeche
        target_face_list = [node for node in list(graph_i.nodes())[-array_total_nodes_each_layer:] if
                             node[2] == Nr_of_Voxel_z - 1]  # Zielflaeche
    
        # Funktion calculate_shortest_path mit MPI auf verschiedenen Noten/CPUs aufrufen
        start = rank * (len(source_nodes_list) // size)
        end = (rank + 1) * (len(source_nodes_list) // size) if rank != size - 1 else len(source_nodes_list)
        local_results = [calculate_shortest_path(graph_i, target_face_list, source_nodes_list[i]) for i in range(start, end)]
        results = comm.gather(local_results, root=0)
            
        # Hautprozess
        if rank == 0:
            path_dict = {}
            average_length = 0
            number_of_path = 0
            shortest_lengths = []            
            
            # Tortuositaet, Anzahl der kuerzesten Pfaden rechnen, die kuerzeste Pfade im dictionary speichern
            for result in results:
                for shortest_length, shortest_path, source_node in result:
                    if shortest_length is not None:
                        path_dict[source_node] = shortest_path
                        average_length += shortest_length
                        shortest_lengths.append(shortest_length/(_Size_of_Voxel_z * (Nr_of_Voxel_z-1)))
                        number_of_path += 1
            try:
                average_length /= number_of_path
            except:
                print(f"Not even one connected path for phase {nr_i}.")
                average_length = 0

            tortuosity_mean = average_length / (_Size_of_Voxel_z * (Nr_of_Voxel_z-1))
            tort_list.append(tortuosity_mean)
            
            if number_of_path > 0:
                tortuosity_std = np.std(np.array(shortest_lengths))                           # standard derivation of tortuosity path
                tortuosity_median = np.median(np.array(shortest_lengths))                     # median value of tortuosity 
            else:
                tortuosity_std = 0
                tortuosity_median = 0
            
            # Ergebnisse in einem Text speichern
            with open(f'{path}Tortuosity_{_fname}_results_SSPSM_Direction{dir}.txt', 'a') as result_Text:
                result_Text.write( '___________________________________\n')
                result_Text.write(f'___________Phase: {nr_i}___________\n')
                result_Text.write(f'Konnektivitaet:{_connectivity}\n')
                result_Text.write(f'Richtung:{dir}\n')
                result_Text.write(f'Gemittelte Tortuositaet: {tortuosity_mean}\n')
                result_Text.write(f'Standardabweichung Tortuositaet: {tortuosity_std}\n')
                result_Text.write(f'Median Tortuositaet: {tortuosity_median}\n\n')
                result_Text.write(f'Länge einzelner Pfade: {shortest_lengths} \n \n')
                result_Text.write(f'Anzahl der verbundenen Pfade:{number_of_path}\n\n')
                result_Text.write('Dictionary:\n')
                for key, value in path_dict.items():
                    result_Text.write(f'{key}:{value}\n')
    
            # Pfade visualisieren
            if _save_paths and number_of_path > 0:
                visualization_of_shortest_path(path_dict)
                np.save(path+f'path_dict_{_fname}_Phase{nr_i}_Direction{dir}.npy', path_dict, allow_pickle=True)
                plt.savefig(f'{path}ShortestPath_{_fname}_SSPSM_{nr_i}_Direction{dir}.pdf')
                plot_tort_distribution(np.array(shortest_lengths), path, f"TortDistribution{_fname}_Phase{nr_i}_Direction{dir}")
                np.save(path+f"shortest_lengths_{_fname}_Phase{nr_i}_Direction{dir}.npy", np.array(shortest_lengths))
        
    return tort_list