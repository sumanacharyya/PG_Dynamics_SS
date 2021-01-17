import numpy as np
import networkx as nx
from Assign_Gen_Load_Power import assign_gen_load_power
# from Assign_Bus_Line_Idx import assign_bus_line_idx
from Compute_Steady_States import compute_steady_states
from Solve_Dynamics import solve_dynamics
# from Compute_Line_Flow import compute_line_flow
from Compute_Theta_Omega import compute_theta_omega
# from Assing_Line_Capacity import assing_line_capacity
# from Check_Line_Overload import check_line_overload
# from Assign_Tstart_Tend import assign_tstart_tend

# from Sort_Lines import sort_lines
# from CutOff_Ovl import cutoff_ovl
# from Cascade_Propagation import cascade_propagation
# from Cascade_Mitigation import cascade_mitigation
# from Cascade_Mitigation_Gi_2 import cascade_mitigation_gi_2
# from Linear_Response_Matrix import compute_Gij_Gij_n
# from network_load import network_load
# from Compute_Gi import compute_Gi
import matplotlib.pyplot as plt
import copy
import random
import time

source = './../../../Configs/'
files = [
         'network_config_RSF_n=6000_m1=3_a=1.0.txt',
         #'network_config_RSF_n=6000_m1=3_a=1.5.txt',
         #'network_config_RSF_n=6000_m1=3_a=2.0.txt'

         ]



def network_info(filename):
    fs = filename[:-4].split('_')[2:]
    s = '_'
    fs = s.join(fs)
    filename = source+filename
    data = eval( open(filename,'r').read() )

    return data

def netx_graph(connections):
    G=nx.Graph()
    for start,links in connections.items():
        for end in links:
            G.add_edge(start, end)
            #G[start][end] = 1.0
           
    return G

def edges_weight(Graph):
    number_edge = 0
    for edge in Graph.edges():
        Graph[edge[0]][edge[1]]['weight'] = 1.0
        
    
def nbrs(Graph):
    nbrS = []
    for node in sorted(Graph.nodes()):
        nbr_i = []
        for nbr in Graph.adj[node]:
            nbr_i.append(nbr)
        nbrS.append(nbr_i)
    
    return nbrS

def incoming_Degree_nodes(Graph):
    Degree_node = []
    for node in sorted(Graph.nodes()):
        income_Degree = 0
        for nbr in Graph.adj[node]:
            income_Degree += Graph[node][nbr]['weight']
        Degree_node.append(income_Degree)
    return Degree_node




def beta_eff_calculation1(DegreesOfNodes,nbrs1):
    b_eff1 = 0
    for i in range(len(nbrs1)):
        nn_weight = 0
        for j in range(len(nbrs1[i])):
            nn_weight += DegreesOfNodes[nbrs1[i][j]]
        b_eff1 += float(nn_weight)/len(nbrs1[i])    
    b_eff1 = b_eff1/len(nbrs1)
    
    return (b_eff1)

def power_jacobian(x,G):
    jaco = (len(x), len(x))
    jaco = np.zeros(jaco)
        
    for node in sorted(G.nodes()):
        sum1 = 0
        for edge in G.adj[node]:
            jaco[node][edge] = K*math.cos(x[edge]- x[node])
            sum1 += math.cos(x[edge]- x[node])
        jaco[node][node] = -1.0*sum1
      
    return jaco

def jaco_file_write(filename, jaco_matrix):
    with open(filename, 'w') as f:
        for row in range(len(jaco_matrix)):
            for column in range(len(jaco_matrix)):
                if(jaco_matrix[row][column]==0):
                    f.write("{:.0f}".format(jaco_matrix[row][column]))
                else:    
                    f.write("{:.10f}".format(jaco_matrix[row][column]))
                f.write('\t')
            f.write('\n')
        f.close()
    



if __name__ == "__main__":
    filename = files[0]
    data = network_info(filename)
    for i,(tag,links) in enumerate(data):
        G = netx_graph(links)
   
    edges_weight(G)
    nbus = len(G)
    ngen = int(len(G)/2)
    nload = nbus - ngen
    d = 2
    K = 10.0  # scaler coupling parameter
    #alpha = 1.2    
    pload = -1.0
    pgen = - (pload*nload)/ngen
    #PGmax = 1.1*pgen
    #PGmin = 0.9*pgen
    gamma = 0.1
    start = time.perf_counter()
    gen_idx, load_idx, power = assign_gen_load_power(pgen, pload, ngen, nload, nbus)
    finish = time.perf_counter()
    print(f'assign_gen_load_power finished in {round(finish - start, 3)} second(s)')
   

    lines = list(G.edges())  # list containing the source and target nodes for all edges
    nlines = G.number_of_edges()
    print(f'number of nodes = {nbus}, number of lines = {nlines}')
    adj = nx.adjacency_matrix(G, nodelist=None, weight=None)
    adj = adj.toarray()
#     bus_idx, line_idx = assign_bus_line_idx(nbus, nlines)

    # Compute Steady States Only
    theta0 = np.random.uniform(10, 11.0, nbus)
    
    
    start = time.perf_counter()
    ss_theta = compute_steady_states(G, power, K, nbus, theta0)
    finish = time.perf_counter()
    print(f'compute_steady_state finished in {round(finish - start, 3)} second(s)')

    Degrees = incoming_Degree_nodes(G)
    nbrs_node = nbrs(G)
    beta_eff = beta_eff_calculation1(Degrees,nbrs_node)

    fs = files[0][:-4].split('_')[2:]
    s = '_'
    fs = s.join(fs)
    Deg_file = f'{fs}_DegNode_Power.txt';
    fd = open(Deg_file, 'w');
    fd.write('#Deg\tNode\tState\n')
    fd.write(f'#Beta_eff={beta_eff}\n')
    for i in range(len(Degrees )):
            fd1.write(f'{Degrees[i]}\t{i}\t{ss_theta[i]}\n')
    fd1.close();

    jaco = power_jacobian(ss_theta,G)
    jaco_file = f'{fs}_jaco_numerical_power.txt'                                             
    jaco_file_write(jaco_file,jaco)
    
    # Testing Steady States
    #y0 = np.zeros(nbus*d)
    #for i in range(nbus):
    #    y0[i*d+0] = ss_theta[i]
    #ti = 0.0; tf = 10; tint = 101
    #tspan, sol_t = solve_dynamics(g0, y0, power, gamma, K, nbus, d, ti, tf, tint)    
    #theta, omega = compute_theta_omega(sol, nbus, d)
    #fig1 = plt.figure(figsize = (8,6))
    #plt.plot(tspan,theta,'-')
    #plt.xlabel('time')
    #plt.ylabel('theta')
    #plt.show()    
    #fig2 = plt.figure(figsize = (8,6))
    #plt.plot(tspan,omega,'-')
    #plt.xlabel('time')
    #plt.ylabel('omega')
    #plt.show()
