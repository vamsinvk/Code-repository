import random
import itertools
import matplotlib.pyplot as plt
import time

class Location:
    def __init__(self, name, x, y, traffic_level, weather_condition):
        self.name = name
        self.x = x
        self.y = y
        self.traffic_level = traffic_level
        self.weather_condition = weather_condition

class RoutePlanner:
    def __init__(self, locations):
        self.locations = locations

    def calculate_distance(self, loc1, loc2):
        return ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5

    def calculate_total_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            distance = self.calculate_distance(route[i], route[i + 1])
            traffic_penalty = route[i].traffic_level + route[i + 1].traffic_level
            weather_penalty = route[i].weather_condition + route[i + 1].weather_condition
            total_distance += distance * (1 + traffic_penalty / 10) * (1 + weather_penalty / 10)
        return total_distance

    def calculate_worst_distance(self, route):
        worst_distance = 0
        for i in range(len(route) - 1):
            distance = self.calculate_distance(route[i], route[i + 1])
            traffic_penalty = max(route[i].traffic_level, route[i + 1].traffic_level)
            weather_penalty = max(route[i].weather_condition, route[i + 1].weather_condition)
            worst_distance += distance * (1 + traffic_penalty / 10) * (1 + weather_penalty / 10)
        return worst_distance

    def find_optimal_and_worst_routes(self):
        locations_list = list(self.locations.values())
        delivery_locations = [loc for loc in locations_list if loc.name.startswith("Delivery")]
        best_route = None
        worst_route = None
        shortest_distance = float("inf")
        worst_distance = 0

        for permutation in itertools.permutations(delivery_locations):
            route = [self.locations["WareHouse"]] + list(permutation) + [self.locations["ReportingOffice"]]
            total_distance = self.calculate_total_distance(route)
            worst_possible_distance = self.calculate_worst_distance(route)

            if total_distance < shortest_distance:
                shortest_distance = total_distance
                best_route = route
            if worst_possible_distance > worst_distance:
                worst_distance = worst_possible_distance
                worst_route = route

        best_route = best_route[0:1] + [loc for loc in best_route[1:-1] if loc not in [self.locations["WareHouse"], self.locations["ReportingOffice"]]] + best_route[-1:]
        worst_route = worst_route[0:1] + [loc for loc in worst_route[1:-1] if loc not in [self.locations["WareHouse"], self.locations["ReportingOffice"]]] + worst_route[-1:]

        return best_route, shortest_distance, worst_route, worst_distance

    def find_greedy_route(self):
        delivery_locations = [loc for loc in self.locations.values() if loc.name.startswith("Delivery")]
        remaining_locations = delivery_locations.copy()
        current_location = self.locations["WareHouse"]
        route = [current_location]

        while remaining_locations:
            nearest_location = None
            min_distance = float('inf')

            for loc in remaining_locations:
                distance = self.calculate_distance(current_location, loc)
                if distance < min_distance:
                    min_distance = distance
                    nearest_location = loc

            route.append(nearest_location)
            remaining_locations.remove(nearest_location)
            current_location = nearest_location

        route.append(self.locations["ReportingOffice"])
        total_distance = self.calculate_total_distance(route)
        return route, total_distance

class GraphicalRepresentation:
    @staticmethod
    def plot_route(route_planner, route, title, distance, route_color):
        x = [loc.x for loc in route_planner.locations.values()]
        y = [loc.y for loc in route_planner.locations.values()]

        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, marker='o', color='g', label='Locations')

        route_x = [loc.x for loc in route]
        route_y = [loc.y for loc in route]

        for i in range(len(route) - 1):
            dist = route_planner.calculate_distance(route[i], route[i + 1])
            plt.annotate(f"Distance: {dist:.2f}", ((route[i].x + route[i + 1].x) / 2, (route[i].y + route[i + 1].y) / 2),
                         textcoords="offset points", xytext=(0, -20), ha='center')

            plt.annotate(f"({route[i].x}, {route[i].y})", (route[i].x, route[i].y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.plot(route_x, route_y, marker='o', linestyle='-', color=route_color, label=title)
        plt.title(f"{title}\nDistance: {distance:.2f}", loc='center')

        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.legend()
        plt.grid(True)

        plt.show()

def generate_random_locations(num_delivery_locations=8):
    random_locations = {}
    random_locations["WareHouse"] = Location("WareHouse", 0, 0, random.randint(1, 3), random.randint(1, 3))
    random_locations["ReportingOffice"] = Location("ReportingOffice", 10, 10, random.randint(1, 3), random.randint(1, 3))

    for i in range(1, num_delivery_locations + 1):
        x = random.randint(0, 10)
        y = random.randint(0, 10)
        location_name = f"Delivery {i}"
        traffic_level = random.randint(1, 3)
        weather_condition = random.randint(1, 3)
        random_locations[location_name] = Location(location_name, x, y, traffic_level, weather_condition)
    return random_locations

# Generate random locations
num_delivery_locations = 8
random_locations = generate_random_locations(num_delivery_locations)

# Initialize route planner
route_planner = RoutePlanner(random_locations)

# Find the greedy route
greedy_route, greedy_distance = route_planner.find_greedy_route()

# Print the greedy route and its distance
print("Greedy Route:", [loc.name for loc in greedy_route])
print("Greedy Distance:", greedy_distance)

# Find the optimal and worst routes
optimal_route, shortest_distance, worst_route, worst_distance = route_planner.find_optimal_and_worst_routes()

# Print the results for the optimal and worst routes
print("Optimal Route:", [loc.name for loc in optimal_route])
print("Shortest Distance:", shortest_distance)
print("Worst Route:", [loc.name for loc in worst_route])
print("Worst Distance:", worst_distance)
print("Execution Time:", end_time - start_time, "seconds")

# Calculate travel times considering traffic intensity and weather conditions
speed = 2  # km per minute (average speed without traffic)

# Calculate time for the optimal route
estimated_time_optimal_route = shortest_distance / (speed * (1 + sum(loc.traffic_level for loc in optimal_route) / 10) * (1 + sum(loc.weather_condition for loc in optimal_route) / 10))

# Calculate time for the worst route
estimated_time_worst_route = worst_distance / (speed * (1 + max(loc.traffic_level for loc in worst_route) / 10) * (1 + max(loc.weather_condition for loc in worst_route) / 10))
print("\nDelivery Locations:")
for loc_name, loc in random_locations.items():
    if loc_name not in ["WareHouse", "ReportingOffice"]:
        print(f"{loc_name}:")
        print(f"   location: ({loc.x}, {loc.y})")
        print(f"   Traffic Level: {loc.traffic_level}")
        print(f"   Weather Condition: {loc.weather_condition}")


# Record the start time before the route computation
start_time = time.time()
optimal_route, shortest_distance, worst_route, worst_distance = route_planner.find_optimal_and_worst_routes()
# Record the end time after the route computation
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")

# Print travel times for the optimal and worst routes
print("Estimated Time for Optimal Route:", estimated_time_optimal_route, "minutes")
print("Estimated Time for Worst Route:", estimated_time_worst_route, "minutes")



# Plot the greedy route graphically
GraphicalRepresentation.plot_route(route_planner, greedy_route, "Greedy Route", greedy_distance, 'blue')


# Plot the optimal and worst routes graphically
GraphicalRepresentation.plot_route(route_planner, optimal_route, "Optimal Route", shortest_distance, 'green')
##GraphicalRepresentation.plot_route(route_planner, worst_route, "Worst Route", worst_distance, 'red')
