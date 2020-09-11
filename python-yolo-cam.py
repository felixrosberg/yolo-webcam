import numpy as np
import cv2
import argparse

def run_yolo(ct, ot, rs):
    LABELS_FILE = "yolo-tiny/coco.names"
    CONFIG_FILE = "yolo-tiny/yolov3-tiny.cfg"
    WEIGHTS_FILE = "yolo-tiny/yolov3-tiny.weights"

    LABELS = open(LABELS_FILE).read().strip().split('\n')

    """PARAMETERS"""
    CONFIDENCE = ct         # Only display objects with more than CONFIDENCE. (0-1)
    THRESHOLD = ot          # decides how much objects are allowed to overlap. (0-1) 1 Being maximum overlap.
    RESOLUTION_SCALE = rs   # cv2 stream the video in a scale of 1440x900. This can be scaled. (0-inf)
    """PARAMETERS"""

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    camera = cv2.VideoCapture(0)
    (W, H) = (None, None)

    while True:
        (return_value, image) = camera.read()

        if W is None or H is None:
            (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Frame", cv2.resize(image, (int(1440 * RESOLUTION_SCALE), int(900 * RESOLUTION_SCALE))))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ct',
                        help='Confidence threshold from 0.0 to 1.0.'
                             'Only display bounding box of objects with'
                             'more than the given confidence.'
                             'Default=0.6',
                        dest='ct',
                        default='0.6')
    parser.add_argument('-ot',
                        help='Overlap threshold from 0.0 to 1.0.'
                             '1 allows maximum overlap, while 0'
                             'allows no overlap.'
                             'Default=0.9',
                        dest='ot',
                        default='0.9')
    parser.add_argument('-rs',
                        help='Resolution scale. The program stream a video of'
                             'resolution 1440x900 at scale 1.'
                             'Default=0.8',
                        dest='rs',
                        default='0.8')
    args = parser.parse_args()
    args = vars(args)

    ct, ot, rs = args['ct'], args['ot'], args['rs']

    run_yolo(ct, ot, rs)


if __name__ == "__main__":
    main()


