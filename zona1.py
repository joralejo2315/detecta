import cv2
import time

# Abre el video
cap = cv2.VideoCapture("/home/pc/detecta-env/detecta/detecta/video/1.mp4")

# Definir la zona de detección (x, y, width, height)
detection_zone = (430, 100, 300, 300)  # Cambia estos valores según tu zona de interés

# Inicializa variables para la diferencia de frames y seguimiento de detección
detection_times = {}
min_area = 500  # Área mínima para considerar un movimiento significativo
movement_threshold = 5

# Variable para contar las veces que se detecta movimiento
movement_count = 0

# Leer los primeros dos frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    if not ret or frame1 is None or frame2 is None:
        print("Video finalizado.")
        break

    # Calcular la diferencia entre los dos frames
    if frame1.shape == frame2.shape:
        diff = cv2.absdiff(frame1, frame2)
    else:
        print("Los frames tienen tamaños diferentes. Saliendo...")
        break

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    current_time = time.time()
    visible_movements = []

    # Dibujar la zona de detección en la imagen
    (zone_x, zone_y, zone_w, zone_h) = detection_zone
    cv2.rectangle(frame1, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), (0, 255, 255), 2)
    cv2.putText(frame1, "Zona de deteccion", (zone_x, zone_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)

        # Verificar si el objeto está dentro de la zona de detección
        if x > zone_x and y > zone_y and (x + w) < (zone_x + zone_w) and (y + h) < (zone_y + zone_h):
            move_id = f"{x}-{y}-{w}-{h}"
            visible_movements.append(move_id)

            if move_id not in detection_times:
                detection_times[move_id] = {'start_time': current_time, 'last_position': (x, y), 'last_move_time': current_time}
                movement_count += 1  # Incrementar el contador al detectar un nuevo movimiento

            else:
                last_x, last_y = detection_times[move_id]['last_position']
                if abs(x - last_x) > movement_threshold or abs(y - last_y) > movement_threshold:
                    detection_times[move_id]['last_move_time'] = current_time
                    detection_times[move_id]['last_position'] = (x, y)

            # Dibujar el rectángulo alrededor del área de movimiento
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Eliminar movimientos que ya no están en vista
    for move_id in list(detection_times.keys()):
        if move_id not in visible_movements:
            del detection_times[move_id]

    cv2.imshow("Video", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

print(f"Movimientos detectados en la zona: {movement_count}")
cap.release()
cv2.destroyAllWindows()
