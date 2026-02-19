import os
import time
import json
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify
# --- Torchvision NMS compatibility patch (Jetson / custom torch builds) ---
# Ultralytics uses torchvision.ops.nms. On Jetson it’s common to have torch with CUDA
# but no matching torchvision build. This patch provides a pure-PyTorch NMS fallback.
import sys, types

def _nms_pytorch(boxes, scores, iou_thres: float):
    """Pure-PyTorch NMS. boxes: (N,4) xyxy, scores: (N,)"""
    import torch
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        inds = (iou <= iou_thres).nonzero(as_tuple=False).squeeze(1)
        order = rest[inds]
    import torch
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

try:
    import torchvision  # type: ignore
    if (not hasattr(torchvision, "ops")) or (not hasattr(torchvision.ops, "nms")):
        class _Ops:  # minimal stub
            pass
        torchvision.ops = _Ops()  # type: ignore
        torchvision.ops.nms = _nms_pytorch  # type: ignore
except Exception:
    torchvision = types.ModuleType("torchvision")
    torchvision.__dict__["__version__"] = "not-installed"
    class _Ops:  # minimal stub
        pass
    torchvision.ops = _Ops()  # type: ignore
    torchvision.ops.nms = _nms_pytorch  # type: ignore
    sys.modules["torchvision"] = torchvision
# --- end patch ---

from ultralytics import YOLO

"""
Web detector with:
- TensorRT/pt model via Ultralytics YOLO
- MJPEG stream in browser
- Save by button:
    * if detection exists and confidence >= SAVE_CONF_MIN -> save JPG + JSON + MASK
    * if workspace clear (no detection) -> save JSON only
- NO periodic status.json writes (status available via /status endpoint in-memory)
- Camera backend:
    * default: try Toupcam (touptek) if installed
    * fallback: OpenCV VideoCapture
"""

# =====================
# CONFIG
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ENGINE = os.path.join(BASE_DIR, "models", "best.engine")
MODEL_PT     = os.path.join(BASE_DIR, "models", "best.pt")

DETECTIONS_DIR = os.path.join(BASE_DIR, "detections")
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# camera backend: "auto" | "toupcam" | "opencv"
CAM_BACKEND = os.environ.get("CAM_BACKEND", "auto").lower()

# OpenCV webcam settings (fallback)
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
CAM_W = int(os.environ.get("CAM_W", "640"))
CAM_H = int(os.environ.get("CAM_H", "480"))

# Toupcam settings (optional)
TOUPCAM_RES_INDEX = int(os.environ.get("TOUPCAM_RES_INDEX", "0"))  # 0 = first supported resolution
TOUPCAM_BITDEPTH = 24  # we use RGB888

# detection
CONF = float(os.environ.get("YOLO_CONF", "0.4"))
SAVE_CONF_MIN = float(os.environ.get("SAVE_CONF_MIN", "0.6"))  # guard against false saves
MIN_CONTOUR_AREA = int(os.environ.get("MIN_CONTOUR_AREA", "300"))

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
    raise FileNotFoundError("No model found: put models/best.engine or models/best.pt in ./models/")

# =====================
# CAMERA BACKENDS
# =====================
class OpenCVCapture:
    def __init__(self, index=0, w=640, h=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class ToupcamCapture:
    """
    Uses Toupcam python module (vendor SDK) with PullMode + callback.
    Keeps only the latest frame in memory (low RAM).
    """
    def __init__(self, res_index=0):
        try:
            import toupcam
        except Exception as e:
            raise RuntimeError(f"toupcam module is not available: {e}")

        self.toupcam = toupcam
        self.hcam = None
        self._lock = threading.Lock()
        self._frame_bgr = None
        self._ok = False

        arr = toupcam.Toupcam.EnumV2()
        if not arr:
            raise RuntimeError("Toupcam: no cameras found (Toupcam.EnumV2() returned empty).")

        self.cur = arr[0]  # first camera
        self.hcam = toupcam.Toupcam.Open(self.cur.id)
        if self.hcam is None:
            raise RuntimeError("Toupcam: failed to open camera.")

        # pick resolution
        try:
            if self.cur.model and self.cur.model.res and len(self.cur.model.res) > 0:
                res_index = max(0, min(res_index, len(self.cur.model.res) - 1))
                self.hcam.put_eSize(res_index)
                w = self.cur.model.res[res_index].width
                h = self.cur.model.res[res_index].height
            else:
                # fallback: query
                w, h = self.hcam.get_Size()
        except Exception:
            w, h = self.hcam.get_Size()

        self.width = int(w)
        self.height = int(h)

        # RGB buffer (bytes) for PullImageV4/V4
        self._buf = bytes(self.toupcam.TDIBWIDTHBYTES(self.width * TOUPCAM_BITDEPTH) * self.height)

        # Start pull mode with callback
        try:
            self.hcam.StartPullModeWithCallback(ToupcamCapture._event_callback, self)
        except Exception as e:
            # Toupcam SDK often raises HRESULTException. One common code is 0x80070005 (Access Denied),
            # typically caused by missing USB permissions (udev rules) or another process holding the camera.
            code = None
            try:
                if hasattr(e, "hr"):
                    code = int(e.hr)
                elif hasattr(e, "hresult"):
                    code = int(e.hresult)
                elif hasattr(e, "args") and len(e.args) > 0:
                    code = int(e.args[0])
            except Exception:
                code = None

            hexcode = None
            if code is not None:
                try:
                    hexcode = hex((code + (1 << 32)) % (1 << 32))
                except Exception:
                    hexcode = None

            msg = f"Toupcam: failed to start streaming callback. HRESULT={code} ({hexcode})"
            print("[ERROR] " + msg, flush=True)

            # Special hint for Access Denied
            if code in (-2147024891, 0x80070005):
                print("[HINT] This looks like ACCESS DENIED (0x80070005). Fix USB permissions with a udev rule.", flush=True)
                print("[HINT] Steps:", flush=True)
                print("  1) Run: lsusb  (find the camera line like 'ID 0547:1361')", flush=True)
                print("  2) Create rule: sudo nano /etc/udev/rules.d/99-toupcam.rules", flush=True)
                print('     Add: SUBSYSTEM=="usb", ATTR{idVendor}=="XXXX", ATTR{idProduct}=="YYYY", MODE="0666"', flush=True)
                print("  3) Apply: sudo udevadm control --reload-rules && sudo udevadm trigger", flush=True)
                print("  4) Unplug/replug the camera USB", flush=True)
                print("[HINT] Also ensure no other app is using the camera.", flush=True)

            raise

        # optional: set auto exposure etc. (leave defaults)

        print(f"[INFO] Toupcam opened: {self.width}x{self.height} (res_index={res_index})", flush=True)

    @staticmethod
    def _event_callback(nEvent, ctx):
        # ctx is self
        try:
            if ctx.hcam and ctx.toupcam.TOUPCAM_EVENT_IMAGE == nEvent:
                ctx._pull_latest()
            elif ctx.hcam and ctx.toupcam.TOUPCAM_EVENT_ERROR == nEvent:
                ctx._ok = False
        except Exception:
            ctx._ok = False

    def _pull_latest(self):
        try:
            # 0=preview image
            self.hcam.PullImageV4(self._buf, 0, TOUPCAM_BITDEPTH, 0, None)
        except Exception:
            self._ok = False
            return

        # buf is RGB888; convert to numpy and then BGR for OpenCV/YOLO
        img = np.frombuffer(self._buf, dtype=np.uint8).reshape((self.height, self.width, 3))
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with self._lock:
            self._frame_bgr = frame_bgr
            self._ok = True

    def read(self):
        with self._lock:
            if self._frame_bgr is None:
                return False, None
            # return a copy-free reference is OK because we overwrite _frame_bgr under lock;
            # still, to be safe for YOLO, return a shallow copy.
            return self._ok, self._frame_bgr.copy()

    def release(self):
        try:
            if self.hcam:
                self.hcam.Close()
        except Exception:
            pass
        self.hcam = None


def create_capture():
    if CAM_BACKEND == "opencv":
        print("[INFO] Camera backend: OpenCV", flush=True)
        return OpenCVCapture(CAM_INDEX, CAM_W, CAM_H)

    if CAM_BACKEND == "toupcam":
        print("[INFO] Camera backend: Toupcam", flush=True)
        return ToupcamCapture(TOUPCAM_RES_INDEX)

    # auto
    if CAM_BACKEND == "auto":
        try:
            cap = ToupcamCapture(TOUPCAM_RES_INDEX)
            print("[INFO] Camera backend: Toupcam (auto)", flush=True)
            return cap
        except Exception as e:
            print(f"[WARN] Toupcam not available, falling back to OpenCV: {e}", flush=True)
            print("[INFO] Camera backend: OpenCV (auto fallback)", flush=True)
            return OpenCVCapture(CAM_INDEX, CAM_W, CAM_H)

    raise ValueError("CAM_BACKEND must be one of: auto, toupcam, opencv")


cap = create_capture()

# =====================
# WEB
# =====================
app = Flask(__name__)

SAVE_REQUESTED = False

LAST_STATUS = {
    "timestamp_ms": 0,
    "object_present": 0,
    "confidence_max": 0.0,
    "bbox_xyxy": None,
    "class_id": None,
    "class_name": None,
}

LAST_ANNOTATED = None
LAST_BEST_BOX = None  # (x1,y1,x2,y2,conf,cls) or None
LAST_GEOM = None      # (grasp_point, axis_endpoints, length_px) or None
LAST_MASK = None      # full mask uint8 or None

# =====================
# UTILS
# =====================
def compute_geometry_and_mask(frame_bgr, bbox_xyxy):
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


def save_event_full(annotated_bgr, best_box, geom, full_mask):
    ts_ms = int(time.time() * 1000)
    stem = os.path.join(DETECTIONS_DIR, f"det_{ts_ms}")

    cv2.imwrite(stem + ".jpg", annotated_bgr)

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

    if full_mask is not None:
        cv2.imwrite(stem + "_mask.png", full_mask)

    return ts_ms


# =====================
# STREAM LOOP
# =====================
def generate():
    global SAVE_REQUESTED, LAST_STATUS
    global LAST_ANNOTATED, LAST_BEST_BOX, LAST_GEOM, LAST_MASK

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        results = model(frame, conf=CONF, verbose=False)
        r = results[0]
        annotated = r.plot()

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

            # geometry + mask
            try:
                grasp, axis, length_px, full_mask = compute_geometry_and_mask(frame, bbox_xyxy)
                geom = (grasp, axis, length_px)

                if axis is not None:
                    (xA, yA), (xB, yB) = axis
                    cv2.line(annotated, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
                if grasp is not None:
                    cv2.circle(annotated, (int(grasp[0]), int(grasp[1])), 6, (0, 255, 255), -1)
            except Exception:
                geom = None
                full_mask = None

        ts_ms = int(time.time() * 1000)
        LAST_STATUS = {
            "timestamp_ms": ts_ms,
            "object_present": int(object_present),
            "confidence_max": float(conf_max),
            "bbox_xyxy": bbox_xyxy,
            "class_id": class_id,
            "class_name": class_name,
        }

        LAST_ANNOTATED = annotated
        LAST_BEST_BOX = best_box
        LAST_GEOM = geom
        LAST_MASK = full_mask

        # save on button
        if SAVE_REQUESTED:
            if LAST_BEST_BOX is None:
                saved_ts = save_event_json_only()
                cv2.putText(annotated, f"JSON-ONLY {saved_ts}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # guard confidence again (just in case)
                if conf_max >= SAVE_CONF_MIN and LAST_ANNOTATED is not None:
                    saved_ts = save_event_full(LAST_ANNOTATED, LAST_BEST_BOX, LAST_GEOM, LAST_MASK)
                    cv2.putText(annotated, f"SAVED {saved_ts}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(annotated, "NOT SAVED: LOW CONF", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            SAVE_REQUESTED = False

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
    return f"""
    <html>
      <head>
        <title>Workspace Monitor</title>
        <style>
          body {{ margin:0; background:#000; overflow:hidden; }}
          #bar {{ position:fixed; top:12px; left:12px; z-index:9999; display:flex; gap:10px; align-items:center; }}
          button {{ font-size:18px; padding:10px 16px; border-radius:10px; border:none; cursor:pointer; }}
          #msg {{ color:#fff; font-family:Arial; font-size:16px; font-weight:700; }}
        </style>
      </head>
      <body>
        <div id="bar">
          <button onclick="saveNow()">Сохранить</button>
          <div id="msg">Порог сохранения: {SAVE_CONF_MIN:.2f}</div>
        </div>

        <img src="/video" style="width:100vw;height:100vh;object-fit:contain;" />

        <script>
          async function saveNow(){{
            const msg = document.getElementById('msg');
            msg.innerText = "Проверяю...";
            try {{
              const r = await fetch('/save', {{method:'POST'}});
              const j = await r.json();
              msg.innerText = j.ok ? ("OK: " + j.message) : (j.message || "ОШИБКА");
            }} catch(e) {{
              msg.innerText = "ERR: " + e;
            }}
          }}
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
    Button logic:
      - no detection -> save JSON-only (workspace clear)
      - detection exists but conf < SAVE_CONF_MIN -> reject
      - detection exists and conf >= SAVE_CONF_MIN -> save full package
    """
    global SAVE_REQUESTED, LAST_BEST_BOX, LAST_STATUS

    if LAST_BEST_BOX is None:
        SAVE_REQUESTED = True
        return jsonify({"ok": True, "message": "WORKSPACE CLEAR: сохраню только JSON (без изображения и маски)."}), 200

    conf_max = float(LAST_STATUS.get("confidence_max") or 0.0)
    if conf_max < SAVE_CONF_MIN:
        return jsonify({"ok": False, "message": f"СЛИШКОМ НИЗКАЯ УВЕРЕННОСТЬ ({conf_max:.2f} < {SAVE_CONF_MIN:.2f})"}), 200

    SAVE_REQUESTED = True
    return jsonify({"ok": True, "message": f"Сохраню кадр + JSON + mask (conf={conf_max:.2f})."}), 200


@app.route("/status")
def status():
    return jsonify(LAST_STATUS)


if __name__ == "__main__":
    print("[INFO] Starting web server on 0.0.0.0:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, threaded=True)
