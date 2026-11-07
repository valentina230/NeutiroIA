"""Física independiente de la red neuronal: salto, obstáculos y colisiones."""

import random

SUELO = 420
JUGADOR_X = 118
JUGADOR_ANCHO = 50
JUGADOR_ALTO = 64
OBSTACULO_ANCHO = 27
GRAVEDAD = 0.65
IMPULSO = -12.0
ANCHO_PISTA = 830  # donde nace el primer obstáculo de cada fila

# Tipos de obstáculo: nombre -> altura en píxeles (más alto = menos margen).
TIPOS_OBSTACULO = {"azul": 36, "rojo": 47, "morado": 58}
ALTURA_MAX = max(TIPOS_OBSTACULO.values())

# Una «fila» tiene 1, 2 o 3 obstáculos seguidos. Entre uno y otro hay tiempo
# de sobra para aterrizar y volver a saltar (separación medida en fotogramas).
TAMANOS_FILA = (1, 2, 3)
PESOS_FILA = (5, 3, 2)
SEPARACION_MIN = 48
SEPARACION_MAX = 70


def _alturas_del_salto():
    """Altura (en px) sobre el suelo en cada fotograma de un salto completo."""
    y, vy, alturas = 0.0, IMPULSO, []
    while True:
        vy += GRAVEDAD
        y += vy
        if y >= 0:
            return alturas
        alturas.append(-y)


_ALTURAS = _alturas_del_salto()


def ventana_ttc(alto, velocidad, margen=2.0):
    """Rango de «fotogramas hasta el obstáculo» en que conviene saltar.

    Depende de la física: hay que estar por encima del obstáculo mientras
    se solapa con Neurito. `margen` deja un colchón de seguridad.
    Devuelve (minimo, maximo), o (None, None) si el salto no alcanza.
    """
    buenos = [i + 1 for i, h in enumerate(_ALTURAS) if h >= alto]
    if not buenos:
        return None, None
    solape = (JUGADOR_ANCHO + OBSTACULO_ANCHO) / max(velocidad, 0.001)
    minimo = buenos[0] - 1 + margen
    maximo = buenos[-1] + 1 - solape - margen
    return minimo, maximo


class JuegoSalto:
    def __init__(self, semilla=19):
        self.azar = random.Random(semilla)
        self.esquivados = 0
        self.choques = 0
        self.reiniciar_pista()

    # --- obstáculos -------------------------------------------------------
    def nueva_fila(self):
        """Crea una fila de 1 a 3 obstáculos (rojo, azul o morado) al azar."""
        cantidad = self.azar.choices(TAMANOS_FILA, weights=PESOS_FILA)[0]
        x = float(ANCHO_PISTA)
        self.obstaculos = []
        for i in range(cantidad):
            tipo = self.azar.choice(list(TIPOS_OBSTACULO))
            self.obstaculos.append(
                {"x": x, "tipo": tipo, "alto": TIPOS_OBSTACULO[tipo]})
            x += self.velocidad * self.azar.uniform(SEPARACION_MIN, SEPARACION_MAX)

    @property
    def tipo(self):
        return self.obstaculos[0]["tipo"]

    @property
    def obstaculo_alto(self):
        return self.obstaculos[0]["alto"]

    @property
    def obstaculo_x(self):
        return self.obstaculos[0]["x"]

    @property
    def pendientes(self):
        """Obstáculos que faltan por pasar en la fila actual."""
        return len(self.obstaculos)

    @property
    def distancia(self):
        """Distancia al obstáculo más cercano que aún no se ha pasado."""
        return max(0.0, self.obstaculo_x - (JUGADOR_X + JUGADOR_ANCHO))

    # --- estado -----------------------------------------------------------
    def reiniciar_pista(self):
        self.y = float(SUELO - JUGADOR_ALTO)
        self.vy = 0.0
        self.en_suelo = True
        self.velocidad = 5.5
        self.terminado = False
        self.pausa_choque = 0
        self.nueva_fila()

    def reiniciar_todo(self):
        self.esquivados = 0
        self.choques = 0
        self.reiniciar_pista()

    def saltar(self):
        if self.en_suelo and not self.terminado:
            self.vy = IMPULSO
            self.en_suelo = False
            return True
        return False

    def actualizar(self):
        if self.terminado:
            self.pausa_choque += 1
            if self.pausa_choque >= 30:
                self.reiniciar_pista()
            return

        if not self.en_suelo:
            self.vy += GRAVEDAD
            self.y += self.vy
            if self.y >= SUELO - JUGADOR_ALTO:
                self.y = float(SUELO - JUGADOR_ALTO)
                self.vy = 0.0
                self.en_suelo = True

        for o in self.obstaculos:
            o["x"] -= self.velocidad
            horizontal = (JUGADOR_X < o["x"] + OBSTACULO_ANCHO
                          and JUGADOR_X + JUGADOR_ANCHO > o["x"])
            vertical = self.y + JUGADOR_ALTO > SUELO - o["alto"]
            if horizontal and vertical:
                self.choques += 1
                self.terminado = True
                return

        if self.obstaculos[0]["x"] + OBSTACULO_ANCHO < JUGADOR_X:
            self.obstaculos.pop(0)
            self.esquivados += 1
            if not self.obstaculos:
                # La velocidad solo cambia entre filas, no dentro de una.
                self.velocidad = self.azar.uniform(4.5, 8.5)
                self.nueva_fila()
