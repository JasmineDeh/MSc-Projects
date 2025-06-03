import numpy as np

def sparrows_hawks_cats(t, y):
    """
    This function calculates how the three species' (sparrows, hawks, and cats) populations change over the time-step, based on birth and death rates.

    Args:
        t (int/float): A single time value.
        y (numpy.ndarray): Parameters to solve. In this function specifically there are three parameters, so this should contain the populations for the species: sparrows, hawks, cats.

    Returns:
        (numpy.ndarray): Change in sparrow, hawk, and cat populations.
    """
    sparrow_population = y[0]
    hawk_population = y[1]
    cat_population = y[2]

    change_in_sparrow = (0.01 * sparrow_population) - (0.001 * sparrow_population * hawk_population) - (0.001 * sparrow_population * cat_population)
    change_in_hawk = (-0.01 * hawk_population) + (0.0001 * sparrow_population * hawk_population)
    change_in_cat = 0.0

    return change_in_sparrow, change_in_hawk, change_in_cat


def sparrows_hawks_cats_2(t, y):
    """
    This function calculates how the three species' (sparrows, hawks, and cats) populations change over the time-step, based on birth and death rates.

    Args:
        t (int/float): A single time value.
        y (numpy.ndarray): Parameters to solve. In this function specifically there are three parameters, so this should contain the populations for the species: sparrows, hawks, cats.

    Returns:
        (numpy.ndarray): Change in sparrow, hawk, and cat populations.
    """
    sparrow_population_2 = y[0]
    hawk_population_2 = y[1]
    cat_population_2 = y[2]

    change_in_sparrow_2 = (0.01 * sparrow_population_2) - (0.001 * sparrow_population_2 * hawk_population_2) - (0.001 * sparrow_population_2 * cat_population_2)
    change_in_hawk_2 = (-0.01 * hawk_population_2) + (0.0001 * sparrow_population_2 * hawk_population_2)
    change_in_cat_2 = (0.0001 * sparrow_population_2  * cat_population_2)


    return change_in_sparrow_2, change_in_hawk_2, change_in_cat_2
    