# Capacitated-Vehicle-Routing-using-ACO
Capacitated Vehicle Routing using Ant Colony Optimization

Vehicle routing problem (VRP) is a combinatorial optimization problem which asks "What is the optimal set of routes for a fleet of vehicles to traverse in order to deliver to a given set of customers?"

### Problem Formulation: 
Each ant is a solution to the problem in the form of a list. The list consists of sublists. Each sublist represents a sequence of locations visited in that route such that the total request quantity for that route (calculated by adding the quantity requested in all the locations of that route) does not exceed the capacity of the truck.

### Results: 
![image](https://user-images.githubusercontent.com/110885397/235709389-10096147-0a78-4e76-bc94-ffcf74294368.png)
Global Best Solution: [[1, 13, 7, 29, 5, 12, 9, 19, 15, 21, 1], [1, 24, 4, 3, 20, 18, 32, 22, 1], [1, 8, 11,
30, 10, 23, 16, 26, 6, 1], [1, 31, 17, 2, 25, 28, 27, 1], [1, 14, 1]]
Fitness: 1052.111757487989
Parameters: no of ants = 20
p = 0.6
Q = 1
alpha = 0.8
beta = 0.8
Total number of iterations = 80

(Please refer to ACO.pdf for detailed report) 
