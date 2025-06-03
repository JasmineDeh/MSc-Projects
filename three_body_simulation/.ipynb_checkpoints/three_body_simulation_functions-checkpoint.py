import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def acceleration(sun_position, earth_position, moon_position, G=6.67E-11, sun_mass=1.98847E30, earth_mass=5.9722E24, moon_mass=7.3476E22):
    """
    Calculates acceleration vectors for the 3 bodies.
    
    Args:
        sun_position (numpy.ndarray): Position of the sun.
        earth_position (numpy.ndarray): Position of the earth.
        moon_position (numpy.ndarray): Position of the moon.
    
    Returns:
        (numpy.ndarray): Array of accelerations.
    """
    sun_earth_diff = earth_position - sun_position
    sun_moon_diff = moon_position - sun_position
    earth_moon_diff = moon_position - earth_position

    sun_earth_mag = np.linalg.norm(sun_earth_diff)
    sun_moon_mag = np.linalg.norm(sun_moon_diff)
    earth_moon_mag = np.linalg.norm(earth_moon_diff)

    sun_earth_unit_pos = sun_earth_diff / sun_earth_mag
    sun_moon_unit_pos = sun_moon_diff / sun_moon_mag
    earth_moon_unit_pos = earth_moon_diff / earth_moon_mag

    sun_earth_force = G * sun_mass * earth_mass / (sun_earth_mag ** 2) * sun_earth_unit_pos
    sun_moon_force = G * sun_mass * moon_mass / (sun_moon_mag ** 2) * sun_moon_unit_pos
    earth_moon_force = G * earth_mass * moon_mass / (earth_moon_mag ** 2) * earth_moon_unit_pos

    sun_force = sun_earth_force + sun_moon_force
    earth_force = -sun_earth_force - -earth_moon_force
    moon_force = -sun_moon_force - earth_moon_force

    sun_acceleration = sun_force / sun_mass
    earth_acceleration = earth_force / earth_mass
    moon_acceleration = moon_force / moon_mass

    return np.array([sun_acceleration, earth_acceleration, moon_acceleration])

def hrs_to_scs(hours):
    """
    Converts hours to seconds.

    Args:
        hours (float): Number of hours.

    Returns:
        (float): Seconds.
    """

    return hours * 3600

def yrs_to_scs(years):
    """
    Converts years to seconds.
    
    Args:
        years (float): Number of years.

    Returns:
        (float): Seconds.
    """

    return years * 365.25 * 24 * 3600

def position(initial_position, velocity, acceleration, time_step):
    """
    Updates the position using the leap-frog scheme.
    
    Args:
        initial_position (numpy.ndarray): Initial position.
        velocity (numpy.ndarray): Velocity of the body.
        acceleration (numpy.ndarray): Acceleration of the body.
        time_step (float): Time step.
   
    Returns:
        (numpy.ndarray): Updated position.
    """
    return initial_position + velocity * time_step + 0.5 * acceleration * (time_step ** 2)

def velocity(previous_velocity, old_acceleration, new_acceleration, time_step):
    """
    Updates the velocity using the leap-frog scheme.
    
    Args:
        previous_velocity (numpy.ndarray): Velocity at the previous step.
        old_acceleration (numpy.ndarray): Acceleration at the previous step.
        new_acceleration (numpy.ndarray): Acceleration at the current step.
        time_step (float): Time step.
   
    Returns:
        (numpy.ndarray): Updated velocity.
    """
    return previous_velocity + 0.5 * time_step * (old_acceleration + new_acceleration)

def valid_data(df, label1, label2, label3):
    """
    Drops the last row on 0.0s on the dataframes, returns numpy array.

    Args:
        df (pandas.DataFrame): DataFrame of interest.
        label1 (string): Column title.
        label2 (string): Column title.

    Returns:
        (numpy.ndarray): Array of valid data.
    """
    validate_positions = df.drop(index=26299)
    valid_positions = np.array(df[[f'x{label1}', f'y{label1}'], [f'x{label2}', f'y{label2}'], [f'x{label3}', f'y{label3}']])

    return valid_positions
    
def orb_eccentricity(sun_position, planet_position):
    """
    Calculates a planet's orbital eccentricity.

    Args:
        sun_position (numpy.ndarray): Sun positions.
        planet_position (numpy.ndarray): Planet positions.

    Returns:
        (float): Eccentricity of the ellipse.
    """
    planet_sun_distance = np.linalg.norm(sun_position - planet_position, axis=1)
    planet_apoapsis = planet_sun_distance.max()
    planet_periapsis = planet_sun_distance.min()

    eccentricity = (planet_apoapsis - planet_periapsis) / (planet_apoapsis + planet_periapsis)

    return eccentricity

def orb_period(moon_position, sun_position, earth_position):
    """
    Calculates the Moon's orbital period.

    Args:
        moon_position (numpy.ndarray): Moon positions.
        sun_position (numpy.ndarray): Sun positions.
        earth_position (numpy.ndarray): Earth positions.

    Returns:
        (numpy.ndarray): 6 months of the Moon's orbital period.
    """
    sun_moon_distance = np.linalg.norm(moon_position - sun_position, axis=1)
    earth_moon_distance = np.linalg.norm(moon_position - earth_position, axis=1)

    moon_differences = earth_moon_distance - sun_moon_distance

    signal = np.array(moon_differences)
    peaks, _ = find_peaks(signal)

    moon_orbital_period = np.linalg.norm(peaks[2] - peaks[1])
    split_periods = np.array_split(moon_differences, 6)

    return peaks, split_periods[0]
    
