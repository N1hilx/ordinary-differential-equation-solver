import numpy as np
import matplotlib.pyplot as plt

def f_pendulum(y, t, eta, omega, a, omega_b):
    """ 
    Vrátí vektor pravé strany diferenciální rovnice tlumeného a buzeného kyvadla.

    Args: 
        y (np.array): Vstupní vektor nezávisle proměnných
        t (float): Čas
        eta, omega, A, omega_b: Doplňující parametry modelu

    Returns:
        np.array: Vektor pravé strany
    
    Examples:
        >>> f_pendulum(np.array([0.5, 0.2]), 12, 0.8, 1.2, 1.5, 0.4)[1] # doctest: +ELLIPSIS
        np.float64(-2.34...)
    """
    
    # Doplňte kód
    
    return np.array([y[1], -eta*y[1] - omega**2 * np.sin(y[0]) + a * np.sin(omega_b * t)])


def runge_kutta_4_step(f, y, t, dt):
    """
    Provede jeden integrační krok metodou Runge-Kutta 4. řádu pro soustavu diferenciálních rovnic 1. řádu.
    
    Args:
        f: Funkce pravých stran f(y, t)
        y (np.array): Vstupní vektor nezávisle proměnných
        t (float): Čas
        dt (float): Časový krok
        
    Returns:
        np.array: Nezávisle proměnné po provedení časového kroku

    Examples:
        >>> runge_kutta_4_step(lambda y, t: np.array([y[1], -y[0]]), np.array([0.0, 1.0]), 0, 0.05)[0] # doctest: +ELLIPSIS
        np.float64(0.049...)
    """

    # Doplňte kód

    k1 = f(y, t)
    k2 = f(y+ k1 * dt/2 , t + dt/2)
    k3 = f(y+ k2 * dt/2 , t + dt/2)
    k4 = f(y+ k3 * dt , t + dt)

    return 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt + y

def ode_solver(f, initial_condition, dt, t_min, t_max, step_function=runge_kutta_4_step):
    """
    Vyřeší soustavu diferenciálních rovnic 1. řádu.
    
    Args:
        f: Funkce pravých stran f(y, t)
        initial_condition (np.array): Vektor počátečních podmínek v čase t_min
        dt (float): Časový krok
        t_min: Počáteční čas integrace
        t_max: Koncový čas integrace
        step_function: Funkce pro integrační krok (touto funkcí lze volit metodu)
        
    Returns:
        np.array: Pole časů a dvojrozměrné pole řešení pro všechny nezávisle proměnné
        
    >>> ode_solver(lambda y, t: np.array([y[1], -y[0]]), np.array([0.0, 1.0]), 0.001, 0, 0.5 * np.pi)[1][-1,0] # doctest: +ELLIPSIS
    np.float64(0.99...)
    >>> ode_solver(lambda y, t: f_pendulum(y, t, 0.5, 1.0, 1.0, 0.2), np.array([0.0, 1.0]), 0.001, 0, 500)[1][-1,0] # doctest: +ELLIPSIS
    np.float64(-0.92...)
    """
    
    t_result = []
    y_result = []

    # Doplňte kód

    t = t_min
    y = initial_condition

    while t < t_max:
        y = step_function(f, y, t, dt)
        t += dt

        t_result.append(t)
        y_result.append(y)

    return np.array(t_result), np.array(y_result)


if __name__ == "__main__":
    eta = 0.5
    omega = 1.0
    a = 1.0
    
    omega_bs = [0.2, 0.4, 0.8]

    dt = 0.01
    t_min, t_max = 0, 500

    ic1 = np.array([0, 1])
    ic2 = np.array([1E-6, 1])
    
    # Vykreslete grafy
    
    for omega_b in omega_bs:
        f = lambda y, t: f_pendulum(y, t, eta, omega, a, omega_b)

        t1, sol1 = ode_solver(f, ic1, dt, t_min, t_max)
        t2, sol2 = ode_solver(f, ic2, dt, t_min, t_max)

        plt.figure()
        plt.plot(t1, sol1[:, 0], label='θ(0) = 0')
        plt.plot(t2, sol2[:, 0], label='θ(0) = 1e-6')
        plt.xlabel('t')
        plt.ylabel('θ(t)')
        plt.title(f'Tlumené a buzené kyvadlo, ω_b = {omega_b}')
        plt.legend()
        plt.grid()

    plt.show()


    # Pro automatické testy odkomentujte
    import doctest
    doctest.testmod(verbose=True) 