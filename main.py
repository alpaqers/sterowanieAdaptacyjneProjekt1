import numpy as np
import matplotlib.pyplot as plt

# Ustawienia
x = np.linspace(0, 6 * np.pi, 1000)  # Zakres od 0 do 6pi
y = np.sin(x)  # Sinusoid

# Generowanie szumu o rozkładzie trójkątnym
noise = np.random.triangular(left=-1, mode=0, right=1, size=x.shape)

# Dodanie szumu do sinusoidy
y_noisy = y + noise

# Głębokość zapominania
def reconstruct_signal(y_noisy, H):
    reconstructed = np.zeros_like(y_noisy)
    reconstructed[0] = y_noisy[0]

    for i in range(1, len(y_noisy)):
        if i <= H:
            reconstructed[i] = sum(y_noisy[:i]) / i
        else:
            reconstructed[i] = sum(y_noisy[i-H:i]) / H
    return reconstructed

# Wartości H do wykresów
H_values = [5, 30, 70, 150, 500]
y_reconstructed = [reconstruct_signal(y_noisy, H) for H in H_values]

# Wykres oryginalnej sinusoidy z szumem
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Sinusoid', color='red', linewidth=1)
plt.scatter(x, y_noisy, label='Noisy Sinusoid', color='blue', s=5, alpha=0.5)
plt.title('Original Sinusoid with Noise')
plt.xlabel('x (radians)')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Wykresy dla różnych wartości H
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='Original Sinusoid', color='red', linewidth=1)
for i, H in enumerate(H_values):
    plt.plot(x, y_reconstructed[i], label=f'H = {H}', linewidth=1)
plt.title('Reconstructed Signals with Different Forgetting Depths (H)')
plt.xlabel('x (radians)')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Obliczanie MSE dla wartości H od 1 do 500
H_range = range(1, 501)
mse_values = []

for H in H_range:
    y_reconstructed_H = reconstruct_signal(y_noisy, H)
    mse = np.mean((y - y_reconstructed_H) ** 2)
    mse_values.append(mse)

# Wykres MSE od H
plt.figure(figsize=(10, 5))
plt.plot(H_range, mse_values, color='purple', linewidth=1.5)
plt.title('Mean Squared Error (MSE) of Reconstructed Signal vs. Forgetting Depth (H)')
plt.xlabel('H')
plt.ylabel('MSE')
plt.grid()
plt.tight_layout()
plt.show()
