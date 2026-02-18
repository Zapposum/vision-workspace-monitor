import os
import time
import json
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ENGINE = os.path.join(BASE_DIR, "models", "best.engine")
MODEL_PT     = os.path.join(BASE_DIR, "models", "best.pt")

# куда сохраняем
DETECTIONS_DIR = os.path.join(BASE_DIR, "detections")
os.makedirs(DETECTIONS_DIR, exist_ok=True)


# камера
CAM_INDEX = 0
CAM_W = 640
CAM_H = 480

# детекция
CONF = 0.4
SAVE_CONF_MIN = 0.6  # минимальная уверенность для сохранения по кнопке
MIN_CONTOUR_AREA = 300

# =====================
# LOAD MODEL
# =====================
if os.path.exists(MODEL_ENGINE):
    print(f"[INFO] Loading TensorRT engine: {MODEL_ENGINE}", flush=True)
    model = YOLO(MODEL_ENGINE)
elif os.path.exists(MODEL_PT):
    print(f"[INFO] Loading PyTorch model: {MODEL_PT}", flush=True)
    model = YOLO(MODEL_PT)
else:
    raise FileNotFoundError("No model found: put models/best.engine or models/best.pt")

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# =====================
# WEB
# =====================
app = Flask(__name__)

# запрос на сохранение по кнопке (обрабатывается в видеопотоке)
SAVE_REQUESTED = False

# кеш последнего статуса (для /status)
LAST_STATUS = {
    "timestamp_ms": 0,
    "object_present": 0,
    "confidence_max": 0.0,
    "bbox_xyxy": None,
    "class_id": None,
    "class_name": None,
}

# кеш последней “лучшей детекции” для сохранения по кнопке
LAST_FRAME = None                # BGR кадр (сырое)
LAST_ANNOTATED = None            # BGR кадр с отрисовкой
LAST_BEST_BOX = None             # (x1,y1,x2,y2,conf,cls) или None
LAST_GEOM = None                 # (grasp_point, axis_endpoints, length_px) или None
LAST_MASK = None                 # full-frame mask uint8 или None


# =====================
# UTILS
# =====================
def atomic_write_json(path: str, payload: dict):
    """Атомарная запись JSON (без битых файлов при чтении)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def compute_geometry_and_mask(frame_bgr, bbox_xyxy):
    """
    bbox_xyxy: [x1,y1,x2,y2] float
    returns:
      grasp_point (x,y) or None
      axis_endpoints ((xA,yA),(xB,yB)) or None
      length_px or None
      full_mask (HxW uint8 0/255) or None
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    H, W = frame_bgr.shape[:2]

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, None, None

    roi = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) > mask.size * 0.6:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, None

    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None, None, None, None

    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), angle = rect
    length = float(max(rw, rh))
    if length <= 1:
        return None, None, None, None

    # угол вдоль длинной оси
    if rw < rh:
        angle += 90.0
    theta = np.deg2rad(angle)

    dx = (length / 2.0) * np.cos(theta)
    dy = (length / 2.0) * np.sin(theta)

    xA, yA = rcx - dx + x1, rcy - dy + y1
    xB, yB = rcx + dx + x1, rcy + dy + y1
    grasp = (rcx + x1, rcy + y1)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    return grasp, ((xA, yA), (xB, yB)), length, full_mask


def save_event_json_only():
    """Сохранение только JSON, когда рабочая зона чистая (workstation clear)."""
    ts_ms = int(time.time() * 1000)
    stem = os.path.join(DETECTIONS_DIR, f"det_{ts_ms}")
    payload = {
        "timestamp_ms": ts_ms,
        "object_present": 0,
        "detection": None,
        "note": "WORKSPACE_CLEAR_JSON_ONLY"
    }
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return ts_ms

def save_event(annotated_bgr, best_box, geom, full_mask):
    """
    Сохраняем по кнопке (при наличии детекции):
      det_<ts>.jpg
      det_<ts>.json
      det_<ts>_mask.png
    """
    ts_ms = int(time.time() * 1000)
    stem = os.path.join(DETECTIONS_DIR, f"det_{ts_ms}")

    # JPG
    cv2.imwrite(stem + ".jpg", annotated_bgr)

    # JSON
    x1, y1, x2, y2, conf, cls = best_box
    class_id = int(cls)

    payload = {
        "timestamp_ms": ts_ms,
        "object_present": 1,
        "detection": {
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "class_id": class_id,
            "class_name": model.names.get(class_id, str(class_id)),
        }
    }

    if geom is not None:
        grasp, axis, length_px = geom
        payload["detection"].update({
            "grasp_point_xy": None if grasp is None else [float(grasp[0]), float(grasp[1])],
            "axis_endpoints_xy": None if axis is None else [
                [float(axis[0][0]), float(axis[0][1])],
                [float(axis[1][0]), float(axis[1][1])]
            ],
            "length_px": None if length_px is None else float(length_px)
        })

    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # MASK
    if full_mask is not None:
        cv2.imwrite(stem + "_mask.png", full_mask)

    return ts_ms


# =====================
# STREAM LOOP
# =====================
def generate():
    global SAVE_REQUESTED, LAST_STATUS
    global LAST_FRAME, LAST_ANNOTATED, LAST_BEST_BOX, LAST_GEOM, LAST_MASK
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        # inference
        results = model(frame, conf=CONF, verbose=False)
        r = results[0]
        annotated = r.plot()

        # defaults: empty workspace
        object_present = 0
        conf_max = 0.0
        bbox_xyxy = None
        class_id = None
        class_name = None

        best_box = None
        geom = None
        full_mask = None

        if r.boxes is not None and len(r.boxes) > 0:
            object_present = 1
            best_i = int(r.boxes.conf.argmax().item())

            conf_max = float(r.boxes.conf[best_i].item())
            class_id = int(r.boxes.cls[best_i].item())
            class_name = model.names.get(class_id, str(class_id))

            x1, y1, x2, y2 = r.boxes.xyxy[best_i].tolist()
            bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
            best_box = (float(x1), float(y1), float(x2), float(y2), float(conf_max), int(class_id))

            # geometry + mask (safe)
            try:
                grasp, axis, length_px, full_mask = compute_geometry_and_mask(frame, bbox_xyxy)
                geom = (grasp, axis, length_px)

                # draw axis + grasp on annotated
                if axis is not None:
                    (xA, yA), (xB, yB) = axis
                    cv2.line(annotated, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
                if grasp is not None:
                    cv2.circle(annotated, (int(grasp[0]), int(grasp[1])), 6, (0, 255, 255), -1)
            except Exception:
                geom = None
                full_mask = None

        # update status cache
        ts_ms = int(time.time() * 1000)
        LAST_STATUS = {
            "timestamp_ms": ts_ms,
            "object_present": int(object_present),  # 0/1
            "confidence_max": float(conf_max),
            "bbox_xyxy": bbox_xyxy,
            "class_id": class_id,
            "class_name": class_name,
        }
        # cache last frame for saving
        LAST_FRAME = frame
        LAST_ANNOTATED = annotated
        LAST_BEST_BOX = best_box
        LAST_GEOM = geom
        LAST_MASK = full_mask
        # if button pressed -> save ONCE:
        #   - if detection exists -> save jpg+json+mask
        #   - else (workspace clear) -> save json-only
        if SAVE_REQUESTED:
            if LAST_BEST_BOX is None:
                saved_ts = save_event_json_only()
                cv2.putText(annotated, f"JSON-ONLY {saved_ts}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                if LAST_ANNOTATED is not None:
                    saved_ts = save_event(LAST_ANNOTATED, LAST_BEST_BOX, LAST_GEOM, LAST_MASK)
                    cv2.putText(annotated, f"SAVED {saved_ts}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            SAVE_REQUESTED = False
        # draw workspace text
        status_text = "WORKSPACE: OCCUPIED" if object_present else "WORKSPACE: CLEAR"
        color = (0, 255, 0) if object_present else (0, 0, 255)
        cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ok2, jpg = cv2.imencode(".jpg", annotated)
        if not ok2:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


# =====================
# ROUTES
# =====================
@app.route("/")
def index():
    # Встроенная кнопка сохранения
    return """
    <html>
      <head>
        <title>Workspace Monitor</title>
        <style>
          body { margin:0; background:#000; overflow:hidden; }
          #bar { position:fixed; top:12px; left:12px; z-index:9999; display:flex; gap:10px; align-items:center; }
          button { font-size:18px; padding:10px 16px; border-radius:10px; border:none; cursor:pointer; }
          #msg { color:#fff; font-family:Arial; font-size:16px; font-weight:700; }
        </style>
      </head>
      <body>
        <div id="bar">
          <button onclick="saveNow()">Сохранить</button>
          <div id="msg">Готово.</div>
        </div>

        <img src="/video" style="width:100vw;height:100vh;object-fit:contain;" />

        <script>
          async function saveNow(){
            const msg = document.getElementById('msg');
            msg.innerText = "Проверяю...";
            try {
              const r = await fetch('/save', {method:'POST'});
              const j = await r.json();
              msg.innerText = j.ok ? ("OK: " + j.message) : (j.message || "ОБЪЕКТ ОТСУТСТВУЕТ");
            } catch(e){
              msg.innerText = "ERR: " + e;
            }
          }
        </script>
      </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/save", methods=["POST"])
def save():
    """
    Логика кнопки:
      - Если детекции НЕТ (WORKSPACE CLEAR) -> сохраним только JSON (без jpg и mask)
      - Если детекция ЕСТЬ, но confidence < SAVE_CONF_MIN -> НЕ сохраняем (защита от ложных)
      - Если детекция ЕСТЬ и confidence >= SAVE_CONF_MIN -> сохраним jpg + json + mask
    """
    global SAVE_REQUESTED, LAST_BEST_BOX, LAST_STATUS

    # workstation clear -> разрешаем сохранение JSON-only
    if LAST_BEST_BOX is None:
        SAVE_REQUESTED = True
        return jsonify({"": True, "message": "WORKSPACE CLEAR: ONLY JSON "}), 200

    conf_max = float(LAST_STATUS.get("confidence_max") or 0.0)
    if conf_max < SAVE_CONF_MIN:
        return jsonify({"": False, "message": f"СЛИШКОМ НИЗКАЯ УВЕРЕННОСТЬ ({conf_max:.2f} < {SAVE_CONF_MIN:.2f})"}), 200

    SAVE_REQUESTED = True
    return jsonify({"": True, "message": f"IMG + JSON + mask (conf={conf_max:.2f})."}), 200


@app.route("/status")
def status():
    return jsonify(LAST_STATUS)


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print("[INFO] Starting web server on 0.0.0.0:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)
