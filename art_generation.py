from PIL import Image, ImageDraw
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)

import imgcompare as ic

# loading in mona_lisa picture
file_location = ''
INPUT_IMAGE = Image.open(file_location)
INPUT_IMAGE.thumbnail((800,800), Image.ANTIALIAS)

# defining some global variables
size = INPUT_IMAGE.size
DIMENSION = size[0]

# Polygon Class generates a single polygon, with an randomized initial color and point locations.
class Polygon:
    
    def __init__(self):
        self.points = random_points()
        self.color = random_color()

    # doing {polygon}.points = set_points() will randomized the points for that said polygon
    def set_points(self):
        self.points = random_points()

    # doing {polygon}.color = set_color() will randomized the color for that said polygon
    def set_color(self):
        self.color = random_color()
    
# Image Class generates a said number of polygons for an easy one line generation of image (image = Img(#polygons))
class Img:

    def __init__(self, num_polygons, fitness_function):
        self.num_polygons = num_polygons
        self.polygons = [Polygon() for _ in range(num_polygons)]
        self.image = draw_polygon(self.polygons)
        self.fitness_function = fitness_function
        self.fitness = 0
  
    def set_image(self):
        self.image = draw_polygon(self.polygons)
    
    def add_polygon(self):
        self.polygons.append(Polygon())
        self.set_image()

    def eval_fitness(self):
        self.fitness = self.fitness_function(self.image)

def fitness(poly_img):
    global INPUT_IMAGE
    return -ic.image_diff_percent(INPUT_IMAGE, poly_img)

#Returns final image with all polygons with Polygon Class
def draw_polygon(population):

    image = Image.new("RGB", (size[0], size[1]), "black")
    draw = ImageDraw.Draw(image, 'RGBA')

    for ind in population:
        split_points = []
        points = ind.points
        for point in points:
            split_points.append(tuple([point[0], point[1]]))
        draw.polygon((split_points[0], split_points[1], split_points[2]), fill=ind.color)

    del draw
    return image

# Generating 3 random points in the dimension of the image
def random_points():
    points = np.array([[-1, -1], [-1, -1], [-1, -1]])
    while ([any(n > 0 for n in lst) for lst in points] == [False, False, False]):
        points[2] = list([random.randrange(0, size[0]), random.randrange(0, size[1])])
        rand = random.random()
        if (rand <= 0.25):
            points[0] = [points[2][0] - DIMENSION / 2, random.randrange(points[2][1], points[2][1] + DIMENSION)]
            points[1] = [points[2][0] + DIMENSION / 2, random.randrange(points[2][1], points[2][1] + DIMENSION)]
        elif (rand <= 0.50):
            points[0] = [random.randrange(points[2][0], points[2][0] + DIMENSION), points[2][1] - DIMENSION / 2]
            points[1] = [random.randrange(points[2][0], points[2][0] + DIMENSION), points[2][1] + DIMENSION / 2]
        elif (rand <= 0.75):
            points[0] = [points[2][0] - DIMENSION / 2, random.randrange(points[2][1] - DIMENSION, points[2][1])]
            points[1] = [points[2][0] + DIMENSION / 2, random.randrange(points[2][1] - DIMENSION, points[2][1])]
        elif (rand <= 1):
            points[0] = [random.randrange(points[2][0] - DIMENSION, points[2][0]), points[2][1] - DIMENSION / 2]
            points[1] = [random.randrange(points[2][0] - DIMENSION, points[2][0]), points[2][1] + DIMENSION / 2]
    return points

# Generating a random color for each polygon, each given a transperancy of 50. Testing will be done on transperancy, some papers had 50-150
def random_color():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    a = 50
    rgba = (r, g, b, a)
    return rgba

def ea_in_order(fitness_function=None, total_generations=100, num_parents=10, num_children=10, num_polygons=10, num_polygons_to_mutate=1, crossover=False, tournament_size=4, num_tournament_winners=2):

    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = -9999 # fitness of best genome so far
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = []

    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Img(num_polygons=1, fitness_function=fitness_function)) # generate new random individuals as parents

     # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness() # evaluate the fitness of each parent

    for generation_num in range(total_generations): # repeat
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            
            # inheretance
            [parent1, parent2] = np.random.choice(population, size=2)
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            # mutation | num_polygons_to_mutate gets set
            # change the color of a polygon
            for this_child in [child1, child2]:
                for _ in range(num_polygons_to_mutate):
                    rand_polygon = np.random.randint(0, len(this_child.polygons))
                    this_child.polygons[rand_polygon].set_color()
                    this_child.polygons[rand_polygon].set_points()
                this_child.set_image()

          
            # crossover
            # swap half of polygons with other child, 2nd half polygons
            if len(population[0].polygons) == num_polygons:
                if crossover:
                    half = int(len(population[0].polygons)/2)
                    temp = child1.polygons[half:]
                    child1.polygons[half:] = child2.polygons[half:]
                    child2.polygons[half:] = temp 
                    child1.set_image()
                    child2.set_image()
                    
            new_children.extend((child1, child2)) # add children to the new_children list
            
        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_fitness()# assign fitness to each child 

        # selection procedure
        population += new_children # combine parents with new children (the + in mu+lambda)
        population = sorted(population, key=lambda img: img.fitness, reverse=True) # sort the full population by each individual's fitness (from highers to lowest)
       

        population = population[:num_parents]

        # record keeping
        if population[0].fitness > solution_fitness: # if the new parent is the best found so far
            solution = population[0]               # update best solution records
            solution_fitness = population[0].fitness

        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time.append(solution)
        
        # start population's num_polygons to be 1, then for the first 50 genreations, we are adding a polygon to the polygon space
        # in each image. This will eliminate noise by forcing the alogirhtm to get a general shape before populating the space. This
        # can be seen as searching the space before exploiting it: exploration.
        if len(population[0].polygons) < num_polygons:
            for ind in population:
                ind.add_polygon()

    return fitness_over_time, solutions_over_time

if __name__ == '__main__':

    total_generations = 250000
    num_polygons = 50
    num_polygons_to_mutate = 1
    num_parents = 50
    num_children = 50
    tournament_size = 10
    num_tournament_winners = 5

    fitness, solution = ea_in_order(fitness_function=fitness, total_generations=total_generations, num_parents=num_parents, num_children=num_children, num_polygons=num_polygons, num_polygons_to_mutate=num_polygons_to_mutate, tournament_size=tournament_size, num_tournament_winners=num_tournament_winners)
    
    plt.plot(fitness)
    plt.xlabel('generation')
    plt.ylabel('Fitness')
    plt.title(f'{total_generations} Generations Fitness Graph')
    plt.savefig('fitness_100k_org.png')


    images = []
    for s in solution:
        images.append(s.image)
    
    images[len(images)-1].save("250k.png")
    images[0].save("ea_in_order_50k.gif", save_all=True, append_images=images[1:], duration=50, loop=0)