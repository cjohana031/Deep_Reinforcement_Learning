# Descripción
Para el reto dos vamos a trabajar en los ambientes de videojuegos clásicos de Box2D y Atari sobre los cuales nos enfocaremos en dos juegos para los cuales se debe crear un agente.

Para resolver las tareas deben escoger dos de los siguientes métodos de aprendizaje por refuerzo profundo

•DQN 

•DQN + experience replay 

•DDQN 

•REINFORCE

# Problema dificultad básica
El primer problema para resolver es Lunar Lander (https://gymnasium.farama.org/environments/box2d/lunar_lander/). Lunar Lander es un ambiente en el que se debe optimizar la trayectoria de una nave espacial. Este ambiente tiene un conjunto discreto de acciones que permiten prender o apagar el motor de la nave. Utilizando la propulsión de los motores, el agente debe configurar una ruta para parquear la nave en la zona designada, entre dos banderas amarillas (siempre ubicadas en la posición (0,0) del ambiente). Igualmente, la nave siempre comienza en la misma posición. El terreno de aterrizaje cambia entre distintas ejecuciones de la nave, sin embargo, la zona de aterrizaje siempre es un área horizontal.

Su tarea para el problema de LunarLander es escoger uno de los métodos de aprendizaje por refuerzo (descritos arriba), y diseñar un agente de aprendizaje por refuerzo profundo que pueda manejar la nave para aterrizar correctamente en la zona de aterrizaje sin importar los cambios en el ambiente. Noten esta tarea se puede desarrollar en el ambiente de colab utilizando laCPU.

Problema dificultad alta
Para el segundo problema a resolver deberán tomar uno de los juegos: 

BattleZone (
https://ale.farama.org/environments/battle_zone/
),

BeamRider (
https://ale.farama.org/environments/beam_rider/
), 

Assault (
https://ale.farama.org/environments/assault/
)

El problema que debe resolver esta dado por: (número de su Team)% 3 + 1

# Problema a resolver

print(f'Problema a resolver: {Team%3 + 1}')
Reset
BattleZone es un juego en el que el agente conduce un tanque el cual debe destruir a los enemigos. El tanque puede moverse en todas las direcciones y debe disparar a los enemigos que se encuentran en el camino, (identificados por un radar). Existen cuatro tipos de enemigos, tanques, supertanques, luchadores y platillos voladores. Destruir cada uno de estos enemigos da un puntaje distinto al tanque. El objetivo del juego es destruir tantos enemigos como sea posible, acumulando la mayor cantidad de puntaje. (Para resolver por los Teams3, 6, 9, 12, ....)

BeamRider es un juego en el que el agente maneja una nave espacial que se mueve hacia adelante con velocidad constante. Las acciones de la nave le permiten moverse lateralmente (derecha e izquierda) y hacia arriba o abajo en estados discretos. Adicionalmente la nave tiene la posibilidad de disparar a las naves enemigas, o usar combinaciones de estas acciones. El objetivo del juego es destruir las naves enemigas sin ser destruidos por ataques enemigos o los asteroides en el espacio. (Para resolver por los Teams 1, 4, 7, 10, ....)

Assault es un juego en el que el agente controla una nave que se puede mover lateralmente (izquierda y derecha). En el juego una nave maestra despliega drones que deben ser destruidos al dispararlesdesde su nave. El objetivo del juego es destruir la mayor cantidad de drones enemigos posible.(Para resolver por los Teams 2, 5, 8, 11, ....)

Su tarea es desarrollar un agente de aprendiza por refuerzo utilizando los métodos descritos arriba. Su agente debe ser capaz de obtener la mayor recompensa promedio en la ejecución de 10 partidas para cada juego. Dadas las características de las imágenes de los juegos, para resolver estos problemas es necesario que utilice la GPU de colab. Note que el entrenamiento para estos ambientes tomará más tiempo del tiempo utilizado en los tutoriales del curso o el problema deLunarLander. Por lo tanto, luego de hacer repetidas evaluaciones de los agentes utilizando la GPU colab puede desconectar su agente, en cuyo caso deberá esperar para poder ejecutar el agente nuevamente. Para aliviar este proceso se recomienda el uso de checkpoints dentro de su ambiente de ejecución.

Problema dificultad avanzada(BONO-competencia)
Como un bono adicional para los estudiantes interesados se presenta un tercer ejercicio a resolver. Este ejercicio utiliza DoubleDunk (
https://ale.farama.org/environments/double_dunk/
), un juego que refleja un partido de basket entre dos equipos de dos jugadores cada uno. El objetivo del juego es ganar el partido. Este ambiente es más complejo que los ambientes anteriores debido a las distintas reglas del juego los movimientos de los agentes y los otros jugadores.

El tiempo de entrenamiento para el agente será significativamente mayor al de los ejemplos utilizados anteriormente en el curso. El objetivo de este ejercicio es poder desarrollar el mejor agente que sea posible con alguno de los métodos descritos arriba, sin que necesariamente el aprendizaje del agente sea óptimo. El bono en esta actividad se otorgará en un formato de competencia, realizando un ranking de los grupos participantes de acuerdo al desempeño de los agentes entregados y con respecto a un agente de base implementado por el equipo docente.

Entrega
La entrega del reto para cada problema debe incluir:

 Se debe entregar el notebook del agente desarrollado en Google colab. El desarrollo de los retos se debe realizar en Google colab. Dentro de la entrega deben enviar el archivo de colab con todas las dependencias para ser ejecutado. Los archivos que no ejecuten tendrán una calificación de 0.

Junto con el notebook de colab con la solución del problema debe entregar el modelo de su agente entrenado, para la verificación de la ejecución.

Debe entregar un video corto demostrando el funcionamiento de los agentes desarrollados, se debe mostrar tanto el funcionamiento del agente en la etapa de entrenamiento como el comportamiento aprendido del agente.

Un reporte profesional (dentro del notebook de su solución) donde expliquen y justifiquen el proceso de desarrollo de los agentes. Este reporte debe tener, por lo menos:

 La decisión de los algoritmos implementados junto con una breve justificación de dicha escogencia. 

Igualmente deben presentar las condiciones de ejecución de su agente proveyendo la información de los parámetros de aprendizaje utilizados, información sobre librerías utilizadas y las características de hardware de la ejecución. 

Además, se debe presentar un análisis de los resultados obtenidos por el agente para cada uno de los problemas resueltos. Su análisis debe estar acompañado de las evidencias del entrenamiento, estadísticas de ejecución y desempeño de los agentes (e.g., tiempo de entrenamiento, puntaje promedio obtenido por el agente en 10 jugadas, graficas de la evolución de la recompensa de entrenamiento y explotación (se recomienda el uso de tensorboard para obtener las gráficas directamente). 

Por último, se debe presentar una conclusión, en base a sus resultados, sobre las capacidades de los agentes implementados (i.e., los métodos) para resolver el problema propuesto


Para el problema del bono únicamente es necesario incluir:

Para cada uno de los problemas a resolver y los métodos utilizados se debe entregar el notebook del agente desarrollado en Google colab. El desarrollo de los retos se debe realizar en Google colab. Dentro de la entrega deben enviar el archivo de colab con todas las dependencias para ser ejecutado. Los archivos que no ejecuten tendrán una calificación de 0.

Junto con el notebook de colab con la solución del problema debe entregar el modelo de su agente entrenado, para la verificación de la ejecución.

Un reporte (en pdf) donde expliquen el algoritmo utilizado y muestre los resultados del agente (i.e., evidencias de entrenamiento, estadísticas de ejecución y desempeño de los agentes (e.g., tiempo de entrenamiento, puntaje promedio obtenido por el agente en 10 jugadas), gráficas de la evolución de la recompensa de entrenamiento y explotación).

Evaluación
La evaluación del reto tendrá en cuenta (para los primeros dos problemas):

La completitud de la entrega. La entrega debe estar completa para cada problema.

La implementación del agente. El notebook de su implementación de su agente debe seguir buenos estándares de programación, estar organizada, debe ser funcional, y debe dar evidencia de un desarrollo dentro del contexto de los métodos utilizados en el curso.

El modelo del agente entrenado debe corresponder al entrenamiento del agente en el notebook.

Para cada uno de los problemas el agente debe ser capaz de resolver el juego efectivamente.

El video debe dar evidencia del proceso de entrenamiento del agente y su ejecución

El reporte debe presentar formalmente el trabajo realizado debidamente justificado.


La evaluación del bono tendrá en cuenta el desempeño del agente para jugar DoubleDunk a modo de competencia. La evaluación se realizará con respecto a una implementación base. El desempeño de los agentes entregados deberá ser por lo menos igual al de la implementación base para ser considerado en el ranking de clasificación del bono. El bono se otorgará de acuerdo a la clasificación.



Licencia

© - Derechos Reservados: La presente obra, y en general todos sus contenidos, se encuentran protegidos por las normas internacionales y nacionales vigentes sobre propiedad Intelectual, por lo tanto su utilización parcial o total, reproducción, comunicación pública, transformación, distribución, alquiler, préstamo público e importación, total o parcial, en todo o en parte, en formato impreso o digital y en cualquier formato conocido o por conocer, se encuentran prohibidos, y solo serán lícitos en la medida en que se cuente con la autorización previa y expresa por escrito de la Universidad de los Andes.


De igual manera, la utilización de la imagen de las personas, docentes o estudiantes, sin su previa autorización está expresamente prohibida. En caso de incumplirse con lo mencionado, se procederá de conformidad con los reglamentos y políticas de la universidad, sin perjuicio de las demás acciones legales aplicables.

Recursos Digitales

Fernando Lozano, Profesor Asociado

Nicolás Cardozo, ProfesorAsociado

Facultad de Ingeniería

Departamento de Ingeniería Eléctrica y Electrónica

Departamento de Ingeniería de Sistemas y Computación

Universidad de los Andes

Bogotá, Colombia

Enero, 2025

