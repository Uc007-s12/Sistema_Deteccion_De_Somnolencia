import cv2
import numpy as np
import time
import mediapipe as mp

class DrowsinessSystem:
    """
    Sistema de detección de somnolencia basado en el Eye Aspect Ratio (EAR).

    Este sistema utiliza MediaPipe Face Mesh para obtener puntos de referencia de los ojos
    y OpenCV DNN para la detección de rostros (recomendado).
    """

    # Índices de los landmarks de MediaPipe para ojos derecho e izquierdo
    _EYE_RIGHT_IDX = [33, 160, 158, 133, 153, 144]
    _EYE_LEFT_IDX  = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        video_path: str = None,
        height: int = 600,
        width: int = 540,
        save_output: bool = False,
        output_path: str = "output.avi",
        camera_index: int = 0,
        use_face_detector: bool = True,
        proto = "deploy.prototxt",
        model = "res10_300x300_ssd_iter_140000.caffemodel",
    ):
        """
        Inicializa el sistema.

        :param video_path: Ruta al archivo de video. Si es None, usa la cámara.
        :param height: Altura del frame para procesar (px).
        :param width: Anchura del frame para procesar (px).
        :param save_output: Si True, guarda la salida en video.
        :param output_path: Ruta de guardado del video de salida.
        :param camera_index: Índice de la cámara en caso de usar captura en vivo o usar algun video.
        :param use_face_detector: Si True, aplica detección de rostros antes de analizar ojos.
        :param proto: Ruta del archivo de configuracion de la red neuronal (prototxt)
        :param model: Ruta al archivo con los pesos preentrenados de la red neuronal (caffemodel) 
        :param _write: Si True, permite que se guarde el video.
        """
        self.video_path = video_path
        self.height = height
        self.width = width
        self.save_output = save_output
        self.output_path = output_path
        self.camera_index = camera_index
        self.use_face_detector = use_face_detector
        self.proto = proto 
        self.model = model
        self._writer = None

        # Inicialización de MediaPipe Face Mesh y detector de rostros
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        if self.use_face_detector:
            self._face_net = cv2.dnn.readNetFromCaffe(self.proto, self.model)
        else:
            self._face_net = None

    def _detect_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Detecta y recorta el rostro usando OpenCV DNN.

        :param frame: Frame original del video.
        :return: Imagen del rostro redimensionada o None.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), swapRB=False, crop=True)
        self._face_net.setInput(blob)
        detections = self._face_net.forward()

        for i in range(detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Limita coordenadas dentro del frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            return cv2.resize(face, (self.width, self.height))
        return None

    def _get_eye_landmarks(self, face: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Obtiene coordenadas de los 6 landmarks de cada ojo.
        Retorna listas de arrays (x, y).
        """
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        right_pts, left_pts = [], []
        if not results.multi_face_landmarks:
            return right_pts, left_pts

        # Solo consideramos el primer rostro
        landmarks = results.multi_face_landmarks[0].landmark
        for idx in self._EYE_RIGHT_IDX:
            lm = landmarks[idx]
            right_pts.append(np.array([int(lm.x * self.width), int(lm.y * self.height)]))
        for idx in self._EYE_LEFT_IDX:
            lm = landmarks[idx]
            left_pts.append(np.array([int(lm.x * self.width), int(lm.y * self.height)]))
        return right_pts, left_pts

    @staticmethod
    def _calculate_ear(eye: list[np.ndarray]) -> float:
        """
        Calcula el Eye Aspect Ratio (EAR).
        """
        # distancias verticales y horizontales
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        h = np.linalg.norm(eye[0] - eye[3])
        return (v1 + v2) / (2.0 * h)

    def calibrate(self, duration: float = 5.0) -> float:
        """
        Calibra el EAR promedio durante un tiempo.

        :param duration: Duración de la calibración en segundos.
        :return: EAR mínimo estimado.
        """
        source = self.video_path or self.camera_index
        cap = cv2.VideoCapture(source)
        ears = []
        start = time.time()

        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            face = self._detect_face(frame) if self.use_face_detector else cv2.resize(frame, (self.width, self.height))
            if face is None:
                cv2.putText(frame, "Face not detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            else:
                r_pts, l_pts = self._get_eye_landmarks(face)
                if r_pts and l_pts:
                    ear = (self._calculate_ear(r_pts) + self._calculate_ear(l_pts)) / 2.0
                    ears.append(ear)
                    cv2.putText(frame, "Calibrando...", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            cv2.imshow("Calibracion", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        if not ears:
            return 0.0
        min_ear = float(min(ears))
        max_ear = float(max(ears))
        print(f"Calibracion finalizada. EAR min: {min_ear:.3f}, max: {max_ear:.3f}")
        return min_ear

    def monitor(
        self,
        threshold: float = 0.22,
        consecutive_secs: float = 0.7
    ) -> None:
        """
        Monitorea el nivel de somnolencia basado en el EAR.

        :param threshold: Umbral EAR para considerar ojos cerrados.
        :param consecutive_secs: Tiempo en segundos que debe mantenerse EAR bajo para alertar.
        """
        source = self.video_path or self.camera_index
        cap = cv2.VideoCapture(source)
        timer = None

        # Configuración para grabar
        if self.save_output:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self._writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = self._detect_face(frame) if self.use_face_detector else cv2.resize(frame, (self.width, self.height))
            if face is not None:
                r_pts, l_pts = self._get_eye_landmarks(face)
                if r_pts and l_pts:
                    ear = (self._calculate_ear(r_pts) + self._calculate_ear(l_pts)) / 2.0
                    # Detecta cierre prolongado
                    if ear < threshold:
                        if timer is None:
                            timer = time.time()
                        elif time.time() - timer >= consecutive_secs:
                            # Esta seccion es en la que se usa la alarma. Agregar el codigo pertinente
                            cv2.putText(frame, "¡Conductor somnoliento detectado!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    else:
                        timer = None
                        cv2.putText(frame, f"EAR: {ear:.2f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            # Guarda y muestra
            if self.save_output and self._writer:
                self._writer.write(frame)
            cv2.imshow("Monitoreo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
