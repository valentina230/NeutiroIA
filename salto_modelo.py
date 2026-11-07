"""Red neuronal 4 -> 12 -> 1 para aprender a saltar, sin frameworks de IA."""
import numpy as np

from salto_juego import ALTURA_MAX, TIPOS_OBSTACULO, ventana_ttc

ENTRADAS = 4


class RedSalto:
    def __init__(self, semilla=42, ocultas=12, tasa=0.8):
        self.rng = np.random.default_rng(semilla)
        self.ocultas = ocultas
        self.tasa = tasa
        self.reiniciar()

    def reiniciar(self):
        self.W1 = self.rng.normal(0, 0.65, (ENTRADAS, self.ocultas))
        self.b1 = np.zeros((1, self.ocultas))
        self.W2 = self.rng.normal(0, 0.65, (self.ocultas, 1))
        self.b2 = np.zeros((1, 1))
        self.epocas = 0
        self.historial = []

    @staticmethod
    def preparar_entradas(distancia, velocidad, alto):
        """Distancia, velocidad, tiempo estimado hasta el obstáculo y su altura."""
        d = np.asarray(distancia, dtype=float)
        v = np.asarray(velocidad, dtype=float)
        a = np.asarray(alto, dtype=float)
        tiempo = np.maximum(d, 0) / np.maximum(v, 0.001)
        return np.column_stack((
            np.clip(d / 550, 0, 1),
            np.clip(v / 9, 0, 1),
            np.clip(tiempo / 60, 0, 1),
            np.clip(a / ALTURA_MAX, 0, 1),
        ))

    @staticmethod
    def sigmoide(z):
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    def adelante(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoide(z2)
        return z1, a1, a2

    def probabilidad(self, distancia, velocidad, alto):
        X = self.preparar_entradas([distancia], [velocidad], [alto])
        return float(self.adelante(X)[2][0, 0])

    @staticmethod
    def etiqueta(distancia, velocidad, alto):
        """Regla que define las etiquetas supervisadas, no una recompensa.

        Saltar si el tiempo hasta el obstáculo cae en la ventana que permite
        la física para ese obstáculo y esa velocidad (ver `ventana_ttc`).
        """
        d = np.atleast_1d(np.asarray(distancia, dtype=float))
        v = np.atleast_1d(np.asarray(velocidad, dtype=float))
        a = np.atleast_1d(np.asarray(alto, dtype=float))
        ttc = d / np.maximum(v, 0.001)
        etiquetas = np.zeros(len(d))
        for i in range(len(d)):
            lo, hi = ventana_ttc(a[i], v[i])
            etiquetas[i] = lo is not None and lo <= ttc[i] <= hi
        return etiquetas.reshape(-1, 1)

    def generar_datos(self, n=5000):
        v = self.rng.uniform(4.5, 8.5, n)
        alto = self.rng.choice(list(TIPOS_OBSTACULO.values()), n).astype(float)
        # Mezcla de ejemplos de saltar y esperar para evitar el desbalance.
        n_pos = n // 2
        ttc = self.rng.uniform(0, 60, n)
        for i in range(n_pos):
            lo, hi = ventana_ttc(alto[i], v[i])
            ttc[i] = self.rng.uniform(lo, hi)
        orden = self.rng.permutation(n)
        v, alto, ttc = v[orden], alto[orden], ttc[orden]
        d = v * ttc
        return self.preparar_entradas(d, v, alto), self.etiqueta(d, v, alto)

    def entrenar_epoca(self, X, y):
        z1, a1, a2 = self.adelante(X)
        p = np.clip(a2, 1e-8, 1 - 1e-8)
        perdida = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Retropropagación de la capa de salida.
        dz2 = (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Retropropagación de la capa oculta (derivada de ReLU).
        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Descenso de gradiente.
        self.W2 -= self.tasa * dW2
        self.b2 -= self.tasa * db2
        self.W1 -= self.tasa * dW1
        self.b1 -= self.tasa * db1
        self.epocas += 1
        self.historial.append(float(perdida))
        return float(perdida)

    def entrenar_varias(self, X, y, n, X_prueba=None, y_prueba=None, tope=None):
        """Entrena hasta `n` épocas; se detiene si la precisión llega al tope.

        Devuelve cuántas épocas entrenó. Con `tope=None` entrena las `n`.
        """
        hechas = 0
        for _ in range(n):
            if tope is not None and self.precision(X_prueba, y_prueba) >= tope:
                break
            self.entrenar_epoca(X, y)
            hechas += 1
        return hechas

    def precision(self, X, y):
        return float(np.mean((self.adelante(X)[2] >= 0.5) == y))
