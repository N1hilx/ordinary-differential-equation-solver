import numpy as np
import matplotlib.pyplot as plt

#dy/dt = sin(ty)
#y_0 = 1
#t_0 = 0

def f(y, t):
    return np.sin(t)*y

def euler_1_step(y, t, dt):
    return y + f(y, t)*dt

def euler_2_step(y, t, dt):
    k1 = f(y, t)
    k2 = f(y + k1*dt, t + dt)
    return y + (k1 + k2)*dt/2

def ode_solver(f, initial_condition, dt, t_min, t_max, step_function=euler_1_step):
    t = t_min
    y = initial_condition

    t_result = [t]
    y_result = [y]

    while t < t_max:
        y = step_function(y, t, dt)
        t += dt

        t_result.append(t)
        y_result.append(y)

    return np.array(t_result), np.array(y_result)

if __name__ == "__main__":
    initial_condition = 1.0
    dt1 = 0.01
    dt2 = 0.01
    t_min = 0.0
    t_max = 10.0

    t1, y1 = ode_solver(f, initial_condition, dt1, t_min, t_max)
    t2, y2 = ode_solver(f, initial_condition, dt2, t_min, t_max, step_function=euler_2_step)

    plt.plot(t1, y1, label=f'dt = {dt1} (Euler 1-step)')
    plt.plot(t2, y2, label=f'dt = {dt2} (Euler 2-step)')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Euler Method for ODE')
    plt.legend()
    plt.grid()
    plt.show()

