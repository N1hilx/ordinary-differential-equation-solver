import numpy as np
import matplotlib.pyplot as plt

#dy/dt = sin(ty)
#y_0 = 1
#t_0 = 0

def f(y, t):
    return np.sin(t)*y


def g(Y, t):
    """ Right-hand side of the Lorenz system of equations. """

    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0

    x, y, z = Y
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z

    return np.array([dxdt, dydt, dzdt])

def euler_1_step(f, y, t, dt):
    return y + f(y, t)*dt

def euler_2_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + k1*dt, t + dt)
    return y + (k1 + k2)*dt/2

def ode_solver(f, initial_condition, dt, t_min, t_max, step_function=euler_1_step):
    t = t_min
    y = initial_condition

    t_result = [t]
    y_result = [y]

    while t < t_max:
        y = step_function(f, y, t, dt)
        t += dt

        t_result.append(t)
        y_result.append(y)

    return np.array(t_result), np.array(y_result)

if __name__ == "__main__":
    initial_condition = np.array([1.0, 1.0, 1.0])
    dt1 = 0.0001
    dt2 = 0.01
    t_min = 0.0
    t_max = 100

    t1, y1 = ode_solver(g, initial_condition, dt1, t_min, t_max, step_function=euler_2_step)
    t2, y2 = ode_solver(g, initial_condition, dt2, t_min, t_max, step_function=euler_2_step)

    plt.plot(y1[:, 0], y1[:, 2], label='Trajectory (XZ projection)', lw=0.5)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Euler Method for ODE')
    plt.legend()
    plt.grid()
    plt.show()

