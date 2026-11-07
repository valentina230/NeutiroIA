
import numpy as np

ACCIONES = ("comer", "dormir", "jugar")


def politica(X):
    """Etiqueta: la necesidad más alta gana. Cansancio = 1 - energía."""
    return np.argmax(np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2])), axis=1)


def relu(v):
    return np.maximum(0, v)


def softmax(v):
    e = np.exp(v - np.max(v, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


class Red:
    def __init__(self, ocultas=6, semilla=42):
        rng = np.random.default_rng(semilla)
        self.W1 = rng.normal(0, 0.5, (3, ocultas))
        self.b1 = np.zeros((1, ocultas))
        self.W2 = rng.normal(0, 0.5, (ocultas, 3))
        self.b2 = np.zeros((1, 3))

    def predecir(self, X):
        return softmax(relu(X @ self.W1 + self.b1) @ self.W2 + self.b2)

    def entrenar(self, X, t, epocas=1200, tasa=0.1):
        n = len(X)
        perdidas = []
        for _ in range(epocas):
            z1 = X @ self.W1 + self.b1
            h = relu(z1)
            p = softmax(h @ self.W2 + self.b2)
            perdidas.append(-np.mean(np.log(p[np.arange(n), t] + 1e-10)))
            d2 = p.copy()
            d2[np.arange(n), t] -= 1
            d2 /= n
            d1 = (d2 @ self.W2.T) * (z1 > 0)
            self.W2 -= tasa * (h.T @ d2)
            self.b2 -= tasa * d2.sum(0, keepdims=True)
            self.W1 -= tasa * (X.T @ d1)
            self.b1 -= tasa * d1.sum(0, keepdims=True)
        return perdidas

    def precision(self, X, t):
        return float(np.mean(np.argmax(self.predecir(X), axis=1) == t))


def datos(semilla):
    rng = np.random.default_rng(semilla)
    X = rng.random((3000, 3))
    Xt = rng.random((1000, 3))
    return X, politica(X), Xt, politica(Xt)


def entrenar(ocultas=6, epocas=1200, tasa=0.1, semilla=42):
    X, t, Xt, tt = datos(semilla)
    r = Red(ocultas, semilla)
    p = r.entrenar(X, t, epocas, tasa)
    return r, p, r.precision(X, t), r.precision(Xt, tt)


if __name__ == "__main__":
    r, _, _, acc = entrenar()
    print(f"Base (6 ocultas, 1200 épocas, tasa 0.1): precisión prueba {acc:.1%}\n")

    print("== 1. Dos necesidades fijas, varía el hambre ==")
    print("energía=0.5 (cansancio 0.5), aburrimiento=0.3")
    previo = None
    for h in np.arange(0, 1.001, 0.05):
        x = np.array([[h, 0.5, 0.3]])
        red = ACCIONES[int(np.argmax(r.predecir(x)))]
        real = ACCIONES[int(politica(x)[0])]
        if red != previo:
            print(f"  hambre={h:.2f} -> red: {red:6s} (regla: {real})")
            previo = red

    print("\n== 2. Prioridades casi iguales: salidas Softmax ==")
    casos = [([0.50, 0.50, 0.10], "hambre 0.50 vs cansancio 0.50"),
             ([0.52, 0.50, 0.10], "hambre 0.52 vs cansancio 0.50"),
             ([0.80, 0.50, 0.10], "hambre 0.80 vs cansancio 0.50"),
             ([0.95, 0.95, 0.95], "hambre 0.95, cansancio 0.05, aburr. 0.95")]
    for x, nombre in casos:
        p = r.predecir(np.array([x]))[0]
        print(f"  {nombre:42s} -> " + "  ".join(
            f"{a}:{v:5.1%}" for a, v in zip(ACCIONES, p)))

    print("\n== 3. Neuronas ocultas (3 semillas, media) ==")
    print("  ocultas | pérdida final | precisión entren. | precisión prueba")
    for n in (3, 6, 10, 20):
        res = [entrenar(ocultas=n, semilla=s) for s in (1, 2, 3)]
        print(f"  {n:7d} | {np.mean([x[1][-1] for x in res]):13.4f} | "
              f"{np.mean([x[2] for x in res]):17.1%} | "
              f"{np.mean([x[3] for x in res]):16.1%}")

    print("\n== 4a. Épocas (tasa 0.1) ==")
    for e in (50, 200, 600, 1200, 5000):
        _, p, _, a = entrenar(epocas=e)
        print(f"  épocas={e:5d} pérdida={p[-1]:.4f} precisión prueba={a:.1%}")

    print("\n== 4b. Tasa de aprendizaje (1200 épocas) ==")
    for lr in (0.001, 0.01, 0.1, 1.0, 5.0, 20.0):
        _, p, _, a = entrenar(tasa=lr)
        osc = int(np.sum(np.diff(p) > 1e-6))
        print(f"  tasa={lr:6.3f} pérdida={p[-1]:.4f} precisión prueba={a:.1%} "
              f"subidas de pérdida={osc}")

    print("\n== 5. Precisión engañosa: clases desbalanceadas ==")
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.05).astype(int)  # 5 % de casos 'urgentes'
    print(f"  Predecir siempre 'no urgente': precisión {np.mean(y == 0):.1%}, "
          f"urgentes detectados 0 de {y.sum()}")
