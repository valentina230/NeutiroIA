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
MORADO = (133, 99, 202)
VERDE = (74, 168, 126)
COLORES_OBSTACULO = {"rojo": (219, 122, 117), "morado": (142, 68, 190)}

# Épocas con las que arranca el cerebro, para que salte desde el principio.
# Con `R` se reinicia sin entrenar y puedes verlo aprender con 1, 2 o 3.
EPOCAS_INICIALES = 3000


def texto(pantalla, fuente, mensaje, x, y, color=TEXTO):
    pantalla.blit(fuente.render(mensaje, True, color), (x, y))


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


def ejecutar():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Neurito 2.0 - Aprende a saltar")
    reloj = pygame.time.Clock()
    fuente = pygame.font.SysFont(None, 27)
    fuente_titulo = pygame.font.SysFont(None, 37)
    fuente_ayuda = pygame.font.SysFont(None, 22)
    mascota = cargar_mascota()

    red = RedSalto()
    juego = JuegoSalto()
    X, y = red.generar_datos(5000)
    X_test, y_test = red.generar_datos(1500)
    automatico = True
    entrenamiento_continuo = False
    en_ejecucion = True

    pantalla.fill(FONDO)
    texto(pantalla, fuente_titulo, "Entrenando el cerebro de Neurito...", 250, 300)
    pygame.display.flip()
    pygame.event.pump()
    for _ in range(EPOCAS_INICIALES):
        red.entrenar_epoca(X, y)

    while en_ejecucion:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                en_ejecucion = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    en_ejecucion = False
                elif evento.key == pygame.K_1:
                    red.entrenar_epoca(X, y)
                elif evento.key == pygame.K_2:
                    for _ in range(100):
                        red.entrenar_epoca(X, y)
                elif evento.key == pygame.K_3:
                    entrenamiento_continuo = not entrenamiento_continuo
                elif evento.key == pygame.K_t:
                    for _ in range(EPOCAS_INICIALES):
                        red.entrenar_epoca(X, y)
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
            red.entrenar_epoca(X, y)

        probabilidad = red.probabilidad(
            juego.distancia, juego.velocidad, juego.obstaculo_alto)
        if automatico and probabilidad >= 0.5 and juego.en_suelo:
            juego.saltar()
        juego.actualizar()

        pantalla.fill(FONDO)
        texto(pantalla, fuente_titulo, "NEURITO 2.0 - APRENDE A SALTAR", 38, 22)
        texto(pantalla, fuente, f"Modo: {'RED NEURONAL' if automatico else 'MANUAL'}", 40, 73)
        texto(pantalla, fuente, f"Épocas: {red.epocas}", 40, 108)
        perdida = f"{red.historial[-1]:.4f}" if red.historial else "sin entrenar"
        texto(pantalla, fuente, f"Pérdida: {perdida}", 235, 108)
        texto(pantalla, fuente, f"Precisión (prueba): {red.precision(X_test, y_test):.1%}", 440, 108)
        texto(pantalla, fuente, f"Probabilidad de salto: {probabilidad:.1%}", 40, 148)
        texto(pantalla, fuente, f"Distancia: {juego.distancia:.0f} px", 355, 148)
        texto(pantalla, fuente, f"Velocidad: {juego.velocidad:.1f} px/frame", 595, 148)

        pygame.draw.rect(pantalla, (230, 246, 238), (32, 194, 894, 247), border_radius=15)
        pygame.draw.line(pantalla, VERDE, (32, SUELO), (926, SUELO), 4)
        obstaculo = pygame.Rect(
            int(juego.obstaculo_x), SUELO - juego.obstaculo_alto,
            OBSTACULO_ANCHO, juego.obstaculo_alto,
        )
        pygame.draw.rect(
            pantalla, COLORES_OBSTACULO[juego.tipo], obstaculo, border_radius=5)
        texto(pantalla, fuente_ayuda,
              f"Obstáculo: {juego.tipo} ({juego.obstaculo_alto} px)", 48, 204)

        rect_mascota = mascota.get_rect(
            center=(JUGADOR_X + 25, int(juego.y) + JUGADOR_ALTO // 2)
        )
        pantalla.blit(mascota, rect_mascota)
        if juego.terminado:
            texto(pantalla, fuente_titulo, "¡CHOQUE!", 420, 270, (186, 64, 80))

        texto(pantalla, fuente, f"Obstáculos esquivados: {juego.esquivados}", 40, 466)
        texto(pantalla, fuente, f"Choques: {juego.choques}", 40, 501)
        texto(pantalla, fuente_ayuda, "1: +1 época     2: +100 épocas     3: entrenamiento continuo", 40, 547)
        texto(pantalla, fuente_ayuda, "A: automático/manual    ESPACIO: salto manual    C: reiniciar pista", 40, 577)
        texto(pantalla, fuente_ayuda, "R: cerebro sin entrenar    T: entrenar 3000 épocas    ESC: salir", 40, 606)
        texto(pantalla, fuente_ayuda, "EVOLUCIÓN DE LA PÉRDIDA", 585, 453)
        dibujar_grafica(pantalla, red.historial)

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()


if __name__ == "__main__":
    ejecutar()
