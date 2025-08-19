# SISTEMA DE MONITOREO DE SOMNOLENCIA BASADO EN MEDIAPIPE Y EYES ASPECT RATIO (EAR). 

El siguiente proyecto se desarrollo durante la estancia académica del 30 Verano de la Investigación Científica y Tecnológica del Pacifico. 

Para realizar la deteccion se usa de un modelo preentrenado de "shiyazt" de su trabajo "Face-Recognition-using-Opencv-: face_detection_model” y Mediapipe. 

Primero se realiza una deteccion y segmentacion del rostro de la persona, se realiza un ajuste del tamaño, despues se obtienen las coordenadas de puntos caracteristicos pertenecientes a los ojos izquiero y derecho, de estos valores se obtiene un promedio de la relacion del aspecto del ojo (EAR), para finalizar se compara con un "umbral EAR", si el valor EAR es menor o igual al umbral, se activa un cronometro. Cuando este cronoemtro es igual o mayor a un valor de segundos consecutivos, se activa una alarma. 

Los valores "umbral EAR" y segundos consecutivos recomendados son 0.22 y 0.7 segundos respectivamente. 

El primer valor se obtuvo al sumar el promedio de los valores EAR de un conjunto de videos de gente con los ojos cerrados. El segundo se obtuvo despues de realizar el analisis de intervalos de tiempo despues de hacer un barrido entre 0.0 segundos hasta 1.0 segundos en intervalor de 0.1 segundos. 

<div align="center">
    <img width="776" height="299" alt="image" src="https://github.com/user-attachments/assets/369262b1-a723-4a4c-9db4-92ebb578d49e" />
</div>
