"""Neurito aprende a saltar: entrenamiento gradual y juego con Pygame."""

from pathlib import Path
import pygame

from salto_juego import (
    JUGADOR_ALTO, JUGADOR_X, OBSTACULO_ANCHO, SUELO, JuegoSalto,
)
from salto_modelo import RedSalto

ANCHO, ALTO = 960, 650
FONDO = (248, 244, 252)
TEXTO = (56, 48, 79)
APAGADO = (116, 106, 146)
BLANCO = (255, 255, 255)
MORADO = (133, 99, 202)
VERDE = (74, 168, 126)
VERDE_OSCURO = (46, 125, 90)
ROJO = (186, 64, 80)
BORDE = (214, 203, 236)
COLORES_OBSTACULO = {
    "azul": (86, 140, 214),
    "rojo": (219, 122, 117),
    "morado": (142, 68, 190),
}

# El cerebro arranca ya entrenado para que Neurito salte desde el principio.
# El entrenamiento se detiene al llegar a esta precisión de prueba.
# Pon None para entrenar sin tope (llega a ~99 % y Neurito no falla).
PRECISION_MAXIMA = 0.94
EPOCAS_INICIALES = 3000  # máximo de épocas al arrancar o al pulsar T


def texto(pantalla, fuente, mensaje, x, y, color=TEXTO):
    imagen = fuente.render(mensaje, True, color)
    pantalla.blit(imagen, (x, y))
    return imagen.get_width()


def tarjeta(pantalla, fuentes, rect, etiqueta, valor, color=TEXTO):
    pygame.draw.rect(pantalla, BLANCO, rect, border_radius=10)
    pygame.draw.rect(pantalla, BORDE, rect, 2, border_radius=10)
    texto(pantalla, fuentes["mini"], etiqueta, rect[0] + 12, rect[1] + 8, APAGADO)
    texto(pantalla, fuentes["valor"], valor, rect[0] + 12, rect[1] + 26, color)


def barra_probabilidad(pantalla, x, y, ancho, alto, p):
    fondo = pygame.Rect(x, y, ancho, alto)
    pygame.draw.rect(pantalla, (233, 227, 244), fondo, border_radius=8)
    relleno = pygame.Rect(x, y, max(0, int(ancho * p)), alto)
    pygame.draw.rect(pantalla, VERDE if p >= 0.5 else MORADO, relleno, border_radius=8)
    pygame.draw.rect(pantalla, BORDE, fondo, 2, border_radius=8)
    medio = x + ancho // 2  # umbral de decisión (50 %)
    pygame.draw.line(pantalla, TEXTO, (medio, y - 3), (medio, y + alto + 3), 2)


def cargar_mascota():
    ruta = Path(__file__).resolve().parent / "assets" / "jugar_1.png"
    imagen = pygame.image.load(str(ruta)).convert_alpha()
    return pygame.transform.smoothscale(imagen, (108, 108))


def dibujar_grafica(pantalla, datos):
    rect = pygame.Rect(570, 478, 345, 104)
    pygame.draw.rect(pantalla, (233, 227, 244), rect, border_radius=9)
    pygame.draw.rect(pantalla, (201, 188, 226), rect, 2, border_radius=9)
    if len(datos) < 2:
        return
    muestras = datos[::max(1, len(datos) // 160)]
    maximo = max(0.2, max(muestras))
    puntos = []
    for i, valor in enumerate(muestras):
        px = rect.left + 9 + i * (rect.width - 18) / max(1, len(muestras) - 1)
        py = rect.bottom - 8 - (valor / maximo) * (rect.height - 16)
        puntos.append((int(px), int(py)))
    pygame.draw.lines(pantalla, MORADO, False, puntos, 3)


def entrenar(red, X, y, X_prueba, y_prueba, epocas):
    return red.entrenar_varias(X, y, epocas, X_prueba, y_prueba, PRECISION_MAXIMA)


def ejecutar():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Neurito 2.0 - Aprende a saltar")
    reloj = pygame.time.Clock()
    fuentes = {
        "titulo": pygame.font.SysFont(None, 37),
        "valor": pygame.font.SysFont(None, 32),
        "normal": pygame.font.SysFont(None, 27),
        "ayuda": pygame.font.SysFont(None, 22),
        "mini": pygame.font.SysFont(None, 20),
    }
    mascota = cargar_mascota()

    red = RedSalto()
    juego = JuegoSalto()
    X, y = red.generar_datos(5000)
    X_test, y_test = red.generar_datos(1500)
    automatico = True
    entrenamiento_continuo = False
    en_ejecucion = True
    desplazamiento = 0.0

    pantalla.fill(FONDO)
    texto(pantalla, fuentes["titulo"], "Entrenando el cerebro de Neurito...", 250, 300)
    pygame.display.flip()
    pygame.event.pump()
    entrenar(red, X, y, X_test, y_test, EPOCAS_INICIALES)

    while en_ejecucion:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                en_ejecucion = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    en_ejecucion = False
                elif evento.key == pygame.K_1:
                    entrenar(red, X, y, X_test, y_test, 1)
                elif evento.key == pygame.K_2:
                    entrenar(red, X, y, X_test, y_test, 100)
                elif evento.key == pygame.K_3:
                    entrenamiento_continuo = not entrenamiento_continuo
                elif evento.key == pygame.K_t:
                    entrenar(red, X, y, X_test, y_test, EPOCAS_INICIALES)
                elif evento.key == pygame.K_a:
                    automatico = not automatico
                elif evento.key == pygame.K_SPACE and not automatico:
                    juego.saltar()
                elif evento.key == pygame.K_c:
                    juego.reiniciar_todo()
                elif evento.key == pygame.K_r:
                    red.reiniciar()
                    juego.reiniciar_todo()
                    entrenamiento_continuo = False

        if entrenamiento_continuo:
            entrenar(red, X, y, X_test, y_test, 1)

        precision = red.precision(X_test, y_test)
        tope = PRECISION_MAXIMA is not None and precision >= PRECISION_MAXIMA
        probabilidad = red.probabilidad(
            juego.distancia, juego.velocidad, juego.obstaculo_alto)
        if automatico and probabilidad >= 0.5 and juego.en_suelo:
            juego.saltar()
        juego.actualizar()
        if not juego.terminado:
            desplazamiento += juego.velocidad

        # --- Dibujo --------------------------------------------------------
        pantalla.fill(FONDO)
        pygame.draw.rect(pantalla, MORADO, (0, 0, ANCHO, 58))
        texto(pantalla, fuentes["titulo"], "NEURITO 2.0 - APRENDE A SALTAR", 32, 17, BLANCO)
        texto(pantalla, fuentes["ayuda"], "Red neuronal 4 -> 12 -> 1", 720, 22, BLANCO)

        perdida = f"{red.historial[-1]:.4f}" if red.historial else "sin entrenar"
        datos_tarjetas = [
            ("MODO", "RED NEURONAL" if automatico else "MANUAL", TEXTO),
            ("ÉPOCAS", str(red.epocas), TEXTO),
            ("PÉRDIDA", perdida, TEXTO),
            ("PRECISIÓN (PRUEBA)", f"{precision:.1%}", VERDE_OSCURO if tope else TEXTO),
        ]
        for i, (etiqueta, valor, color) in enumerate(datos_tarjetas):
            tarjeta(pantalla, fuentes, (32 + i * 228, 72, 210, 58), etiqueta, valor, color)

        texto(pantalla, fuentes["normal"], "Probabilidad de salto", 32, 148)
        barra_probabilidad(pantalla, 270, 149, 420, 20, probabilidad)
        texto(pantalla, fuentes["normal"], f"{probabilidad:.1%}", 704, 148)
        salta = probabilidad >= 0.5
        texto(pantalla, fuentes["normal"], "SALTAR" if salta else "ESPERAR", 805, 148,
              VERDE_OSCURO if salta else APAGADO)

        pygame.draw.rect(pantalla, (230, 246, 238), (32, 184, 894, 257), border_radius=15)
        x = 48
        x += texto(pantalla, fuentes["ayuda"], f"Distancia: {juego.distancia:.0f} px", x, 194) + 26
        x += texto(pantalla, fuentes["ayuda"], f"Velocidad: {juego.velocidad:.1f}", x, 194) + 26
        color_sig = COLORES_OBSTACULO[juego.tipo]
        pygame.draw.rect(pantalla, color_sig, (x, 195, 14, 14), border_radius=3)
        x += 20
        x += texto(pantalla, fuentes["ayuda"],
                   f"Siguiente: {juego.tipo} ({juego.obstaculo_alto} px)", x, 194) + 26
        texto(pantalla, fuentes["ayuda"], f"En fila: {juego.pendientes}", x, 194)

        pygame.draw.line(pantalla, VERDE, (32, SUELO), (926, SUELO), 4)
        for i in range(0, 894, 60):
            marca = 926 - ((i + desplazamiento) % 894)
            if marca + 16 <= 926:
                pygame.draw.line(pantalla, (160, 205, 180),
                                 (int(marca), SUELO + 10), (int(marca) + 16, SUELO + 10), 3)

        for o in juego.obstaculos:
            ox = int(o["x"])
            if ox > 926 - OBSTACULO_ANCHO:
                continue  # todavía no entra en pantalla
            rect = pygame.Rect(ox, SUELO - o["alto"], OBSTACULO_ANCHO, o["alto"])
            color = COLORES_OBSTACULO[o["tipo"]]
            pygame.draw.rect(pantalla, color, rect, border_radius=5)
            oscuro = tuple(int(c * 0.7) for c in color)
            pygame.draw.rect(pantalla, oscuro, rect, 2, border_radius=5)

        altura = SUELO - (juego.y + JUGADOR_ALTO)
        ancho_sombra = int(max(20, 60 - altura * 0.3))
        sombra = pygame.Rect(0, 0, ancho_sombra, 10)
        sombra.center = (JUGADOR_X + 25, SUELO + 4)
        pygame.draw.ellipse(pantalla, (190, 220, 203), sombra)

        rect_mascota = mascota.get_rect(
            center=(JUGADOR_X + 25, int(juego.y) + JUGADOR_ALTO // 2)
        )
        pantalla.blit(mascota, rect_mascota)
        if juego.terminado:
            aviso = fuentes["titulo"].render("¡CHOQUE!", True, ROJO)
            pantalla.blit(aviso, (32 + 894 // 2 - aviso.get_width() // 2, 290))

        tarjeta(pantalla, fuentes, (32, 456, 150, 58), "ESQUIVADOS",
                str(juego.esquivados), VERDE_OSCURO)
        tarjeta(pantalla, fuentes, (196, 456, 150, 58), "CHOQUES",
                str(juego.choques), ROJO)
        if tope:
            texto(pantalla, fuentes["ayuda"],
                  f"Tope de precisión ({PRECISION_MAXIMA:.0%}) alcanzado: entrenamiento detenido",
                  40, 524, VERDE_OSCURO)
        texto(pantalla, fuentes["ayuda"], "1: +1 época     2: +100 épocas     3: entrenamiento continuo", 40, 550)
        texto(pantalla, fuentes["ayuda"], "A: automático/manual    ESPACIO: salto manual    C: reiniciar pista", 40, 580)
        texto(pantalla, fuentes["ayuda"], "R: sin entrenar    T: entrenar al máximo    ESC: salir", 40, 610)
        texto(pantalla, fuentes["ayuda"], "EVOLUCIÓN DE LA PÉRDIDA", 585, 453)
        dibujar_grafica(pantalla, red.historial)

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()


if __name__ == "__main__":
    ejecutar()
