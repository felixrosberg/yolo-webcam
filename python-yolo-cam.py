import numpy as np
import cv2

LABELS_FILE = "yolo-tiny/coco.names"
CONFIG_FILE = "yolo-tiny/yolov3-tiny.cfg"
WEIGHTS_FILE = "yolo-tiny/yolov3-tiny.weights"

LABELS = open(LABELS_FILE).read().strip().split('\n')

CONFIDENCE = 0.3
THRESHOLD = 0.9
RESOLUTION_SCALE = 0.7

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

camera = cv2.VideoCapture(0)
(W, H) = (None, None)
writer = None

while True:
    (return_value, image) = camera.read()

    if W is None or H is None:
        (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_output = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]],  confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  2)

    cv2.imshow("Frame",  cv2.resize(image, (int(1440) * RESOLUTION_SCALE, int(700) * RESOLUTION_SCALE)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()

        


