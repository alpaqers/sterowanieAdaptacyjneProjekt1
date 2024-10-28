import numpy as np
import matplotlib.pyplot as plt

# Ustawienia
x = np.linspace(0, 6 * np.pi, 1000)  # Zakres od 0 do 2pi
y = np.sin(x)  # Sinusoid

# Generowanie szumu o rozkładzie trójkątnym
noise = np.random.triangular(left=-1, mode=0, right=1, size=x.shape)

# Dodanie szumu do sinusoidy
y_noisy = y + noise

# Głębokość zapominania
H = 30

# Wyciąganie średniej z punktów o głębokości H
def reconstruct_signal(y_noisy, H):
    reconstructed = np.zeros_like(y_noisy)
    reconstructed[0] = y_noisy[0]

    for i in range(1, len(y_noisy)):
        reconstructed[i] = sum(y_noisy[i-(H):i])/H

    return reconstructed

y_reconstructed_5 = reconstruct_signal(y_noisy,5)
y_reconstructed_30 = reconstruct_signal(y_noisy,30)
y_reconstructed_70 = reconstruct_signal(y_noisy,70)
y_reconstructed_150 = reconstruct_signal(y_noisy,150)
y_reconstructed_500 = reconstruct_signal(y_noisy,500)

# Rysowanie wykresu
plt.plot(x, y_noisy, label='Noisy Sinusoid', color='blue', linewidth=3)
plt.plot(x, y_reconstructed_5, label='Reconstructed Signal', color='orange', linewidth=1)
plt.plot(x, y_reconstructed_30, label='Reconstructed Signal', color='yellow', linewidth=1)
plt.plot(x, y_reconstructed_70, label='Reconstructed Signal', color='green', linewidth=1)
plt.plot(x, y_reconstructed_150, label='Reconstructed Signal', color='black', linewidth=1)
plt.plot(x, y_reconstructed_500, label='Reconstructed Signal', color='black', linewidth=1)
plt.title('Reconstructed Signal with Forgetting')
plt.xlabel('x (radians)')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
