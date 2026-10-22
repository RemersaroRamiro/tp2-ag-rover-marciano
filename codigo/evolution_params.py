"""
evolution_params.py: las constantes del experimento evolutivo.

Para qué existe
    Reúne en un solo lugar las decisiones del experimento que el informe tiene que declarar. No son
    medidas físicas del rover (eso está en rover_params) ni reglas de la simulación (eso está en
    evaluate.EvalConfig).

Qué contiene
    - Aptitud: INFEASIBLE_OFFSET = 1.0. Un diseño inválido puntúa -(1.0 + violación total), siempre por
      debajo de cualquier diseño válido, que puntúa su distancia (≥ 0). La violación total es la suma
      de los excesos relativos de las restricciones no cumplidas: (valor - límite) / límite. Cualquier
      valor positivo da el mismo orden; 1.0 solo mantiene legibles los puntajes. Reemplaza a una
      penalización anterior de 500 m por unidad de violación, que a los mejores diseños les convenía
      pagar: quedaban entre un 4 % y un 7 % fuera del volumen permitido.
    - Semillas: TERRAIN_SEED_LIMIT = 2**31 - 1 (las semillas de terreno se eligen en [1, 2**31 - 1)) y
      SCHEDULE_TAG, que se mezcla con la semilla de la corrida para que las semillas de terreno salgan
      de un flujo aleatorio distinto del que usa el AG.
    - Validación y éxito: los VALIDATION_GENOMES = 3 mejores genomas se validan en VALIDATION_SEEDS = 5
      terrenos nunca vistos. Una corrida tiene éxito si su mejor diseño es válido y llega a la meta en
      SUCCESS_MIN_FINISHES = 3 de SUCCESS_OF_SEEDS = 5 terrenos (entrega parcial, sección 6.3); con otra
      cantidad de terrenos se exige la misma proporción, redondeada hacia arriba.
    - Grabación: REPLAY_HZ = 10 cuadros por segundo simulado; REPLAY_MAX_FRAMES = 6000 cuadros por
      rover como máximo (un rover que llega a la meta necesita unos 5900, así que solo se reduce la
      tasa de uno que avanza muy lento hasta el tiempo límite); REPLAY_BEST_K = 5 rovers por generación
      en el modo de grabación "best".

Dónde encaja
    Etapa: aptitud, corrida y resultados. No importa otros archivos del proyecto.
    Lo usan fitness, evolve, seed_schedule y replay.
"""

from __future__ import annotations

INFEASIBLE_OFFSET = 1.0

TERRAIN_SEED_LIMIT = 2**31 - 1
SCHEDULE_TAG = 0x52_4F_56_33
VALIDATION_SEEDS = 5
VALIDATION_GENOMES = 3

SUCCESS_MIN_FINISHES = 3
SUCCESS_OF_SEEDS = 5

REPLAY_HZ = 10.0
REPLAY_MAX_FRAMES = 6000
REPLAY_BEST_K = 5
