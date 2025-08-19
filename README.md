# SISTEMA DE MONITOREO DE SOMNOLENCIA BASADO EN MEDIAPIPE Y EYES ASPECT RATIO (EAR). 

El siguiente proyecto se desarrolló durante la estancia académica del 30 Verano de la Investigación Científica y Tecnológica del Pacifico (Programa Delfin). 

Para realizar la detección se usa de un modelo preentrenado de "Shiyazt" de su trabajo "Face-Recognition-using-Opencv-: face_detection_model”, y las librerías de numpy, Mediapipe, CV2, y time. 

Primero se realiza una detección y segmentación del rostro de la persona, se realiza un ajuste del tamaño, después se obtienen las coordenadas de puntos característicos pertenecientes a los ojos izquierdo y derecho, de estos valores se obtiene un promedio de la relación del aspecto del ojo (EAR), para finalizar se compara con un "umbral EAR", si el valor EAR es menor o igual al umbral, se activa un cronometro. Cuando este cronómetro es igual o mayor a un valor de segundos consecutivos, se activa una alarma. 
<br>
    <div align="center">
        <img width="776" height="600" alt="Imagen del conductor" src="https://github.com/user-attachments/assets/631595a2-fafe-4df5-aa1d-dd1c3d72a6c1" />
    </div>
<br>

Los valores "umbral EAR" y "segundos consecutivos" recomendados son 0.22 y 0.7 segundos respectivamente. 

El primer valor se obtuvo al sumar el promedio de los valores EAR de un conjunto de videos de gente con los ojos cerrados. 
El segundo se obtuvo después de realizar el análisis de intervalos de tiempo en videos del sistema funcionando, haciendo un barrido entre 0.0 segundos hasta 1.0 segundos en intervalo de 0.1 segundos. La mayor parte del tiempo del proyecto fue dedicado a esta parte y se obtuvieron los siguientes valores.

<br>
    <div align="center">
        <img width="776" height="299" alt="image" src="https://github.com/user-attachments/assets/369262b1-a723-4a4c-9db4-92ebb578d49e" />
    </div>
<br>

Por lo que lo recomendable es tener un intervalo entre 0.7 segundos hasta 1.0 segundos. 

## Conclusión.
Este sistema tiene como debilidad vestimenta que use el usuario como lentes oscuros, además de la hora del día, pues no realiza un buen trabajo para lugares obscuros, igualmente se deseó realizar la comparación usando un modelo reentrenado de YoLo para la detección de ojos abiertos y cerrados, pero debido a falta de tiempo no se logró terminar el entrenamiento.

