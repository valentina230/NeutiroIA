"""Física independiente de la red neuronal: salto, obstáculos y colisiones."""

import random

SUELO = 420
JUGADOR_X = 118
JUGADOR_ANCHO = 50
JUGADOR_ALTO = 64
OBSTACULO_ANCHO = 27
GRAVEDAD = 0.65
IMPULSO = -12.0

# Tipos de obstáculo: nombre -> altura en píxeles.
# El morado es más alto, así que deja menos margen para saltar.
TIPOS_OBSTACULO = {"rojo": 47, "morado": 58}
ALTURA_MAX = max(TIPOS_OBSTACULO.values())


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

    def nuevo_obstaculo(self):
        """Coloca un obstáculo al azar (rojo o morado) al borde derecho."""
        self.tipo = self.azar.choice(list(TIPOS_OBSTACULO))
        self.obstaculo_alto = TIPOS_OBSTACULO[self.tipo]
        self.obstaculo_x = 830.0

    def reiniciar_pista(self):
        self.y = float(SUELO - JUGADOR_ALTO)
        self.vy = 0.0
        self.en_suelo = True
        self.velocidad = 5.5
        self.terminado = False
        self.pausa_choque = 0
        self.nuevo_obstaculo()

    def reiniciar_todo(self):
        self.esquivados = 0
        self.choques = 0
        self.reiniciar_pista()

    @property
    def distancia(self):
        return max(0.0, self.obstaculo_x - (JUGADOR_X + JUGADOR_ANCHO))

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

        self.obstaculo_x -= self.velocidad
        horizontal = (JUGADOR_X < self.obstaculo_x + OBSTACULO_ANCHO
                      and JUGADOR_X + JUGADOR_ANCHO > self.obstaculo_x)
        vertical = (self.y + JUGADOR_ALTO > SUELO - self.obstaculo_alto)
        if horizontal and vertical:
            self.choques += 1
            self.terminado = True
            return

        if self.obstaculo_x + OBSTACULO_ANCHO < JUGADOR_X:
            self.esquivados += 1
            self.nuevo_obstaculo()
            self.velocidad = self.azar.uniform(4.5, 8.5)
