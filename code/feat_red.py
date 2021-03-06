import os
os.chdir('C:/Users/krish/OneDrive/Krishna/Coursework/STAT578/final_project/FeatureReductionGenetic-master/FeatureReductionGenetic-master')

import numpy
import GA
import pickle
import matplotlib.pyplot
import pandas as pd
import numpy as np



dataset_names = ['DPP4','HIVINT', 'HIVPROT','METAB','OX1']

save_root='C:/Users/krish/Desktop/STAT 578/data/processed/'

best_ind_root='C:/Users/krish/Desktop/STAT 578/data/Best_indices/'


for i in range (2,len(dataset_names)):
    data=pd.read_csv(save_root+dataset_names[i]+'_train_processed.csv')
    data.set_index('MOLECULE', inplace=True)

    data_inputs=data.loc[:, data.columns != 'Act'].values
    data_outputs=data.loc[:,data.columns=='Act'].values.ravel()
    #data_outputs=np.reshape(data_outputs,(-1,1))


    num_samples = data_inputs.shape[0]
    num_feature_elements = data_inputs.shape[1]

    train_indices = numpy.arange(1, num_samples, 4)
    test_indices = numpy.arange(0, num_samples, 4)
    print("Number of training samples: ", train_indices.shape[0])
    print("Number of test samples: ", test_indices.shape[0])
    print("Current dataset:",dataset_names[i])
    """
    Genetic algorithm parameters:
        Population size
        Mating pool size
        Number of mutations
        """
    sol_per_pop = 8 # Population size.
    num_parents_mating = 4 # Number of parents inside the mating pool.
    num_mutations = 3 # Number of elements to mutate.

    # Defining the population shape.
    pop_shape = (sol_per_pop, num_feature_elements)

    # Creating the initial population.
    new_population = numpy.random.randint(low=0, high=2, size=pop_shape)
    print(new_population.shape)

    best_outputs = []
    num_generations = 25
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Measuring the fitness of each chromosome in the population.
        fitness = GA.cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)

        best_outputs.append(numpy.max(fitness))
        # The best result in the current iteration.
        #print("Best result : ", best_outputs[-1])
        
        # Selecting the best parents in the population for mating.
        parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)
        
        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))
        
        # Adding some variations to the offspring using mutation.
        offspring_mutation = GA.mutation(offspring_crossover, num_mutations=num_mutations)
        
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        
        # Getting the best solution after iterating finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        fitness = GA.cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))[0]
        best_match_idx = best_match_idx[0]

    best_solution = new_population[best_match_idx, :]
    best_solution_indices = numpy.where(best_solution == 1)[0]
    best_solution_num_elements = best_solution_indices.shape[0]
    best_solution_fitness = fitness[best_match_idx]

    print("best_match_idx : ", best_match_idx)
    print("best_solution : ", best_solution)
    print("Selected indices : ", best_solution_indices)
    print("Number of selected elements : ", best_solution_num_elements)
    print("Best solution fitness : ", best_solution_fitness)

    matplotlib.pyplot.plot(best_outputs)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.show()
    #best_solution.to_csv(best_ind_root+dataset_names[i]+'_best_ind.csv')
    np.savetxt(best_ind_root+dataset_names[i]+'_best_ind.csv', best_solution.astype(int), delimiter=",")
