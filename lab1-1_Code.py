import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([155, 160, 165, 170, 175, 180, 185, 190])
y_train = np.array([55, 60, 65, 70, 75, 80, 85, 90])

w_init = 0.5
b_init = -25
alpha = 0.0000001
iterations = 10000

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    return (1 / (2 * m)) * cost_sum

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    return dj_dw / m, dj_db / m

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i < 100000:
            J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

w_final, b_final, J_hist = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)
print(f"Знайдені параметри (w,b): ({w_final:.4f}, {b_final:.4f})")

plt.scatter(x_train, y_train, marker='x', c='r', label='Реальні дані')
plt.plot(x_train, w_final * x_train + b_final, 'b-', label='Прогноз моделі')
plt.xlabel('Зріст (см)')
plt.ylabel('Вага (кг)')
plt.title('Лінійна регресія: Зріст vs Вага')
plt.legend()
plt.grid(True)
plt.show()

alphas_to_test = [1e-8, 1e-7, 4.3e-7] 
iters_test = 50

plt.figure(figsize=(10, 6))

for test_alpha in alphas_to_test:
    _, _, J_hist = gradient_descent(x_train, y_train, w_init, b_init, test_alpha, iters_test)
    
    plt.plot(J_hist, label=f'alpha = {test_alpha}')

plt.xlabel('Кількість ітерацій')
plt.ylabel('Вартість (J)')
plt.title('Вплив швидкості навчання (alpha) на збіжність алгоритму')
plt.legend()
plt.grid(True)
plt.show()

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

y_pred = w_final * x_train + b_final
print(f"Коефіцієнт детермінації (R²): {r_squared(y_train, y_pred):.4f}")