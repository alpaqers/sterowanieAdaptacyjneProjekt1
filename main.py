import numpy as np
import matplotlib.pyplot as plt


# Funkcja do rekonstrukcji sygnału
def reconstruct_signal(y_noisy, H):
    reconstructed = np.zeros_like(y_noisy)
    reconstructed[0] = y_noisy[0]

    for i in range(1, len(y_noisy)):
        if i <= H:
            reconstructed[i] = sum(y_noisy[:i]) / i
        else:
            reconstructed[i] = sum(y_noisy[i - H:i]) / H
    return reconstructed


# Ustawienia
x = np.linspace(0, 6 * np.pi, 1000)  # Zakres od 0 do 6pi
y = np.sin(x)  # Sinusoid

# Wzrost wariancji
variance_values = np.arange(0, 2.6, 0.1)  # Wariancje od 0 do 2.5
optimal_H_values = []
mse_values = []

# Iteracja przez wartości wariancji
for variance in variance_values:
    # Obliczanie parametrów rozkładu trójkątnego
    left = -np.sqrt(24 * variance)  # Oblicz left
    right = np.sqrt(24 * variance)  # Oblicz right
    mode = 0  # Ustawienie mod na 0

    # Upewnij się, że left i right są różne
    if left == right:
        right += 0.01  # Dodać małą wartość do right, aby były różne

    # Generowanie szumu
    noise = np.random.triangular(left=left, mode=mode, right=right, size=x.shape)

    # Dodanie szumu do sinusoidy
    y_noisy = y + noise

    # Obliczanie MSE dla różnych H
    H_range = range(1, 50)
    lowest_mse = float('inf')
    Optimal_H = 0

    for H in H_range:
        y_reconstructed_H = reconstruct_signal(y_noisy, H)
        mse = np.mean((y - y_reconstructed_H) ** 2)
        if mse < lowest_mse:
            lowest_mse = mse
            Optimal_H = H

    optimal_H_values.append(Optimal_H)
    mse_values.append(lowest_mse)  # Przechowuj MSE dla tej wariancji

# Wykres zależności optymalnego H od wariancji
plt.figure(figsize=(10, 5))
plt.plot(variance_values, optimal_H_values, marker='o', color='blue', linewidth=1.5)
plt.title('Optimal H vs Variance of Noisy Signal')
plt.xlabel('Variance')
plt.ylabel('Optimal H')
plt.grid()
plt.tight_layout()
plt.show()

# Wykres MSE od wariancji
plt.figure(figsize=(10, 5))
plt.plot(variance_values, mse_values, marker='x', color='purple', linewidth=1.5)
plt.title('Mean Squared Error (MSE) vs Variance of Noisy Signal')
plt.xlabel('Variance')
plt.ylabel('MSE')
plt.grid()
plt.tight_layout()
plt.show()
