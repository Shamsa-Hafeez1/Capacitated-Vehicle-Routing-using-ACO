import xml.etree.ElementTree as ET 
import random 
import numpy as np 
import copy 
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

# p is the gamma value 

class ACO: 
    def __init__(self, file_name, no_of_ants, p, Q, alpha, beta): 
        print("initialized.....")
        tree = ET.parse(file_name)
        self.no_of_ants = no_of_ants
        self.p = p
        root = tree.getroot() 

        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.locations  = [(int(float(node.find('cx').text)), int(float(node.find('cy').text))) for node in root.findall('.//node')]

        self.requests = [] # node, quantity 

        for request in root.findall('.//request'):
            node = int(request.get('node'))
            quantity = int(float(request.find('quantity').text))
            self.requests.append((node, quantity))

        self.map_ = dict() 
        self.tau = dict() 
        self.departure_node = int(float(root.find('.//departure_node').text))  
        self.arrival_node = int(float(root.find('.//arrival_node').text))  
        self.capacity = int(float(root.find('.//capacity').text))  
        self.make_map()  # make the map of the graph formed 
    
    def distance(self, a, b): # calculated distance between a and b 
        return ((a[1] - b[1]) ** 2  + (a[0] - b[0]) ** 2 ) ** 0.5
    
    def make_map(self): 
        '''
        Self: Reference to the class 

        Updates the map such that it makes the upper triangle (In matrix notation, it has the upper triangular values)

        Returns: None 
        '''
        x = 0

        for i in range(len(self.locations)): 

            for j in range(x, len(self.locations)): 
            
                src = self.locations[i] 
                dst = self.locations[j]

                self.tau[(src, dst)] = 0 
                self.map_[(src, dst)] = [0, self.distance(src, dst)] # pheramne, distance between two towns 

            x += 1 

    def initialize_ants(self):
        ants = [] 
        for i in range(self.no_of_ants): 
            ants.append(self.initialize_ant(copy.deepcopy(self.requests)))
        return ants
    
    def initialize_ant(self, requests): 

        ant = [] 
      
        while len(requests) != 0: 
            route = [self.departure_node] 
            cap = 0 
            request = copy.deepcopy(requests)

            while cap < self.capacity and len(request) != 0: 
                customer = random.choice(request) # CHANGING pas route[-1], request 
                request.remove(customer) 

                if customer[1] + cap <= self.capacity: 
                    requests.remove(customer) # that node is visited 
                    cap += customer[1]        
                    route.append(customer[0])

            route.append(self.arrival_node)
            ant.append(route)
            
        return ant 
    
    def regenerate_ant(self): 
        '''
        Args: 
        - self

        Returns: 
        - List of new ants based on probabilities 
        '''
        ant = [] 
        for coord in self.map_: 
            t = ((self.tau[coord]) ** self.alpha)
            try: 
                n = ((1 / self.map_[coord][1]) ** self.beta) # division by zero 
            except: 
                n = 0  
            self.map_[coord][0] =  t * n 
      
        rqs = copy.deepcopy(self.requests) # making a deep copy since we just want to rule out all possibilities  

        while len(rqs) != 0: 
            route = [self.departure_node] 
            cap = 0 
            request = copy.deepcopy(rqs)

            while cap < self.capacity and len(request) != 0: 


                probability = self.computing_p(node = route[-1], possible_dest = request)
                
                if all(x == 0 for x in probability):
                    customer = random.choice(request)
                else: 
                    values = np.array([i for i in range(len(request))])
                    custom_dist = rv_discrete(values=(values, probability)) 
                    index = custom_dist.rvs() 
                    customer = tuple(request[index])

                request.remove(customer) 

                if customer[1] + cap <= self.capacity: 
                    rqs.remove(customer) # that node is visited 
                    cap += customer[1]        
                    route.append(customer[0])

            route.append(self.arrival_node)
            ant.append(route) # new route in the total path 
            
        return ant 

    def computing_p(self, node, possible_dest):  #node = route[-1], possible_dest = request)
        '''
        Args: 
        - self: 
        - node: The source node / location from which we can move forward 
        - possible_dest: List of places to where my ant can go, standing at node 

        Returns: 
        List of probabilities of going to possible_dest 
        '''
        source = self.locations[node-1]
        
        p = []
        for i in possible_dest: 
            if (self.locations[i[0]-1], source) in self.map_:
                p.append(self.map_[(self.locations[i[0]-1], source)][0]) 
            else: 
                p.append(self.map_[(source, self.locations[i[0]-1])][0]) 
        
        return [0 if i == 0 else i / sum(p) for i in p]
    
    def combine_one_ant(self, ant): 
        '''
        Consolidates the path of an ant: 
        For example: From [[1, 2, 3, 1], [1,4,1]] it becomes [1, 2, 3, 1, 4, 1]
        This is just to simplify future computations in this code logic 
        '''
        lst = []  
        for i in ant[:-1]:
            lst.extend(i[:-1]) 
        
        lst.extend(ant[-1])
        return lst 
    
    def all_ants_distance(self, ants):
        '''
        Args: 
        - ants : List of ants of size no_of_ants

        Returns: 
        - a list with Q / distance covered by ant 
        '''
        return [self.Q / self.total_distance_per_ant(ant) for ant in ants] 
    
    def total_distance_per_ant(self, ant):
        '''
        Args: 
        - self: Reference to class 
        - ant: The ant (list) of which total distance needs to be calculated 

        Returns: 
        - Total distance covered by the ant i.e., total distance in all the routes 
        '''
        total_route = 0 
        for route in ant: 

            for i in range(len(route)-1): 
        
                # The map has upper triangular matrix so seeing what key is present in the map_ 
                if (self.locations[route[i]-1], self.locations[route[i+1]-1]) in self.map_: 
                    total_route += self.map_[(self.locations[route[i]-1], self.locations[route[i+1]-1])][1]
                else: 
                    total_route += self.map_[(self.locations[route[i+1]-1], self.locations[route[i]-1])][1]
            
        return total_route 
    
    def find_tau(self, ants): 
        '''
        Args: 
        - self: Reference to the class
        - ants: New ants on the basis of which tau table shall be updated 

        This function updates the ta
        Returns: 
        - None 
        '''
        updated = {key: False for key in self.tau}
        all_ants_dist = self.all_ants_distance(ants)
        ants = [self.combine_one_ant(ant) for ant in ants] 
       
        for x in range(len(ants)): 
            ant = ants[x]
    
            for i in range(len(ant)-1): 
                src = ant[i]
                dst = ant[i+1] 

                src_t = self.locations[src-1]
                dst_t = self.locations[dst-1]

                if ((dst_t, src_t) in self.tau and not updated[(dst_t, src_t)]) : 
                        src_t, dst_t = dst_t, src_t
                        src, dst = dst, src 

                if ((src_t, dst_t) in self.tau and not updated[(src_t, dst_t)]):
                    self.tau[(src_t, dst_t)] += all_ants_dist[x]
                    updated[(src_t, dst_t)] = True 

                    if src == 1:
                        try: 
                            if ant[i+2] == src: 
                                self.tau[(src_t, dst_t)] += all_ants_dist[x]
                        except: 
                            pass 
                
                        for j in range(x+1,len(ants)): 
                            indices = list(np.where( np.asarray(ants[j]) == src)[0])
                            indices.remove(len(ants[j]) - 1) 

                            for y in indices: 
                            
                                if ants[j][y+1] == dst: # cater to situations like 1, 5, 1
                                    self.tau[(src_t, dst_t)] += all_ants_dist[j]

                                if ants[j][y-1] == dst:
                                    self.tau[(src_t, dst_t)] += all_ants_dist[j]
                            
                    else: 
                        for j in range(x+1,len(ants)): 
                            try: 
                                y = ants[j].index(src)
                                if ants[j][y+1] == dst: 
                                    self.tau[(src_t, dst_t)] += all_ants_dist[j]
                                if ants[j][y-1] == dst:
                                    self.tau[(src_t, dst_t)] += all_ants_dist[j]
                            except: 
                                pass 
                                    
                else: 
                    pass 
    

    def update_tau(self, ants):
        '''
        Args:  
        ants: The new ants on the basis of which tau table needs to be updated 

        Returns: 
        None 
        '''
        for i in self.tau: 
            self.tau[i] *= (1-self.p)
        self.find_tau(ants)


    def total_distance_per_ant(self, ant):
        '''
        Returns total route distance covered by the ant 
        '''
        total_route = 0 
        for route in ant: 
            for i in range(len(route)-1): 
                if (self.locations[route[i]-1], self.locations[route[i+1]-1]) in self.map_: 
                    total_route += self.map_[(self.locations[route[i]-1], self.locations[route[i+1]-1])][1]
                else: 
                    total_route += self.map_[(self.locations[route[i+1]-1], self.locations[route[i]-1])][1]

        return total_route 

    

filename = ["A-n32-k05.xml", "A-n44-k06.xml", "A-n60-k09.xml", "A-n80-k10.xml"] 
print(filename)
indx = int(input("Enter the file number: 0/1/2/3 :"))
no_of_ants = int(input("Enter the number of ants:"))
aco = ACO(file_name = filename[indx], no_of_ants = no_of_ants, p = 0.6, Q = 1, alpha = 2, beta = 3)
ants = aco.initialize_ants() 
aco.update_tau(ants)

best_fitness = [] 
avg_fitness = [] 
best_solution = None 
best_fitness_of_best_sol = None 
x = [] 

num_of_iterations = 100
for i in range(num_of_iterations): 
    print(i)
    x.append(i)
    ants = [aco.regenerate_ant() for i in range(no_of_ants)] 
    aco.update_tau(ants)

    lst = [ aco.total_distance_per_ant(i) for i in ants]
    min_val = min(lst) 
    best_fitness.append(min_val)

    if min_val <= min(best_fitness): 
        best_solution = ants[lst.index(min_val)]
        best_fitness_of_best_sol = min_val

    avg_fitness.append(sum(lst) / len(lst))

print("Best solution: ", best_solution)
print("Fitness of best solution: ",best_fitness_of_best_sol)
fig, ax = plt.subplots()
ax.plot(x, best_fitness, '-o', label='Best Fitness')
ax.plot(x, avg_fitness, '-o',  label='Average Fitness')
ax.set_title(filename[indx])
ax.set_xlabel('Iteration No.')
ax.set_ylabel('Fitness')
ax.legend()
plt.show()
    
