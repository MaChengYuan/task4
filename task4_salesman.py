import sys
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

class State:
    # Create a new state
    def __init__(self, route:[], distance:int=0):
        self.route = route
        self.distance = distance
    # Compare states
    def __eq__(self, other):
        for i in range(len(self.route)):
            if(self.route[i] != other.route[i]):
                return False
        return True
    # Sort states
    def __lt__(self, other):
         return self.distance < other.distance
    # Print a state
    def __repr__(self):
        return ('({0},{1})\n'.format(self.route, self.distance))
    # Create a shallow copy
    def copy(self):
        return State(self.route, self.distance)
    # Create a deep copy
    def deepcopy(self):
        return State(copy.deepcopy(self.route), copy.deepcopy(self.distance))
    # Update distance
    def update_distance(self, matrix, home):
        
        # Reset distance
        self.distance = 0
        # Keep track of departing city
        from_index = home
        # Loop all cities in the current route
        for i in range(len(self.route)):
            self.distance += matrix[from_index][self.route[i]]
            from_index = self.route[i]
        # Add the distance back to home
        self.distance += matrix[from_index][home]
# This class represent a city (used when we need to delete cities)
class City:
    # Create a new city
    def __init__(self, index:int, distance:int):
        self.index = index
        self.distance = distance
    # Sort cities
    def __lt__(self, other):
         return self.distance < other.distance
# Return true with probability p
def probability(p):
    return p > random.uniform(0.0, 1.0)
# Schedule function for simulated annealing
def exp_schedule(k=20, lam=0.005, limit=1000):
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)
# Get the best random solution from a population
def get_random_solution(matrix:[], home:int, city_indexes:[], size:int, use_weights:bool=False):
    # Create a list with city indexes
    cities = city_indexes.copy()
    # Remove the home city
    cities.pop(home)
    # Create a population
    population = []
    for i in range(size):
        if(use_weights == True):
            state = get_random_solution_with_weights(matrix, home)
        else:
            # Shuffle cities at random
            random.shuffle(cities)
            # Create a state
            state = State(cities[:])
            state.update_distance(matrix, home)
        # Add an individual to the population
        population.append(state)
    # Sort population
    population.sort()

    return population[0]

def get_best_solution_by_distance(matrix:[], home:int):
    
    # Variables
    route = []
    from_index = home
    length = len(matrix) - 1
    iteration = 0
    # Loop until route is complete
    while len(route) < length:
        iteration+=1

        row = matrix[from_index]

        cities = {}
        for i in range(len(row)):
            cities[i] = City(i, row[i])
        # Remove cities that already is assigned to the route
        del cities[home]
        for i in route:
            del cities[i]
        # Sort cities
        sorted = list(cities.values())
        sorted.sort()
        # Add the city with the shortest distance
        from_index = sorted[0].index
        route.append(from_index)
    # Create a new state and update the distance
    state = State(route)
    state.update_distance(matrix, home)
    print('iteration : {}'.format(iteration))
    # Return a state
    return state

def get_random_solution_with_weights(matrix:[], home:int):
    

    route = []
    from_index = home
    length = len(matrix) - 1
    # Loop until route is complete
    while len(route) < length:
         # Get a matrix row
        row = matrix[from_index]

        cities = {}
        for i in range(len(row)):
            cities[i] = City(i, row[i])
        # Remove cities that already is assigned to the route
        del cities[home]
        for i in route:
            del cities[i]
        # Get the total weight
        total_weight = 0
        for key, city in cities.items():
            total_weight += city.distance

        weights = []
        for key, city in cities.items():
            weights.append(total_weight / city.distance)
        # Add a city at random
        from_index = random.choices(list(cities.keys()), weights=weights)[0]
        route.append(from_index)
    # Create a new state and update the distance
    state = State(route)
    state.update_distance(matrix, home)

    return state

def mutate(matrix:[], home:int, state:State, mutation_rate:float=0.01):
    
    # Create a copy of the state
    mutated_state = state.deepcopy()
    # Loop all the states in a route
    for i in range(len(mutated_state.route)):
        # Check if we should do a mutation
        if(random.random() < mutation_rate):
            # Swap two cities
            j = int(random.random() * len(state.route))
            city_1 = mutated_state.route[i]
            city_2 = mutated_state.route[j]
            mutated_state.route[i] = city_2
            mutated_state.route[j] = city_1
    # Update the distance
    mutated_state.update_distance(matrix, home)

    return mutated_state

def simulated_annealing(matrix:[], home:int, initial_state:State, mutation_rate:float=0.01, schedule=exp_schedule()):
    # Keep track of the best state
    best_state = initial_state
    # Loop a large number of times (int.max)
    for t in range(sys.maxsize):
        # Get a temperature
        T = schedule(t)
        # Return if temperature is 0
        if T == 0:
            return best_state
        # Mutate the best state
        neighbor = mutate(matrix, home, best_state, mutation_rate)
        # Calculate the change in e
        delta_e = best_state.distance - neighbor.distance
        # Check if we should update the best state
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            best_state = neighbor

def paint_shortest_plot(path,city_point,label):
    x=[]
    y=[]
    for i in range(0,len(path)):
        x.append(city_point[path[i]][0])
        y.append(city_point[path[i]][1])
    plt.plot(x,y,label=label)
    plt.legend()

    return

    
def main():

    cities = ['New York', 'Los Angeles', 'Chicago', 'Minneapolis', 'Denver', 'Dallas', 'Seattle', 'Boston', 'San Francisco', 'St. Louis', 'Houston', 'Phoenix', 'Salt Lake City','taiwan','russia']
    city_indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    home = 2 # Chicago
    # Distances in miles between cities, same indexes (i, j) as in the cities array
    city_dot ='  0.549963E-07  0.985808E-08\
      -28.8733     -0.797739E-07 \
      -79.2916      -21.4033    \
      -14.6577      -43.3896    \
      -64.7473       21.8982    \
      -29.0585      -43.2167    \
      -72.0785      0.181581    \
      -36.0366      -21.6135    \
      -50.4808       7.37447    \
      -50.5859      -21.5882    \
     -0.135819      -28.7293    \
      -65.0866      -36.0625    \
      -21.4983       7.31942    \
      -57.5687      -43.2506    \
      -43.0700       14.5548    '
    

    
    list_of_strings = city_dot.split()
    list_of_strings = np.array(list_of_strings)
    x =list_of_strings.reshape(15,2)
    x = x.astype(float)

    
    for i in range(x.shape[0]):
        plt.scatter(x[i][0],x[i][1])

    
    
    
    
    matrix = np.ones((x.shape[0],x.shape[0]))
    
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            matrix[i][j] = matrix[j][i] = np.sqrt((x[i][0]-x[j][0])**2+(x[i][1]-x[j][1])**2)



    state = get_random_solution(matrix, home, city_indexes, 100)
    print('-- random solution --')
    print(cities[home], end='')
    for i in range(0, len(state.route)):
        print(' -> ' + cities[state.route[i]], end='')
    print(' -> ' + cities[home], end='')
    print('\n\nTotal distance: {0} miles'.format(state.distance))
    print()
    label = 'initial path'
    random_array = state.route
    random_array.insert(0,home)
    random_array.insert(len(random_array),home)
    paint_shortest_plot(random_array,x,label)

    state = get_best_solution_by_distance(matrix, home)
    state = simulated_annealing(matrix, home, state, 0.1)
    ##################################
    label = 'sorted path'
    array = state.route
    array.insert(0,home)
    array.insert(len(array),home)
    paint_shortest_plot(array,x,label)
   #################################

    print('-- Simulated annealing solution --')
    print(cities[home], end='')
    for i in range(0, len(state.route)):
       print(' -> ' + cities[state.route[i]], end='')
    print(' -> ' + cities[home], end='')
    print('\n\nTotal distance: {0} miles'.format(state.distance))
    print()

if __name__ == "__main__": main()