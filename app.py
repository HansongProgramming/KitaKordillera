import cv2
import numpy as np
import json
import argparse
import math
import datetime
from dataclasses import dataclass, asdict

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("This script requires 'mediapipe'. Install with: pip install mediapipe opencv-python numpy")


@dataclass
class Finding:
    name: str
    present: bool
    confidence: float  # 0-1 heuristic confidence
    reasoning: str


@dataclass
class Report:
    timestamp: str
    image_path: str
    notes: str
    findings: list


# MediaPipe landmark indices for eyes/iris (FaceMesh with refine_landmarks=True adds iris landmarks 468-477)
# We'll define polygons for the left/right eye region using canonical mesh indices.
LEFT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # rough boundary
RIGHT_EYE_LANDMARKS = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

LEFT_IRIS_LMS = [468, 469, 470, 471, 472]
RIGHT_IRIS_LMS = [473, 474, 475, 476, 477]


def polygon_mask(shape, points):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def l2(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def circle_from_points(pts):
    # Fit circle via least squares to iris landmarks (x,y). Returns center (x,y) and radius.
    pts = np.array(pts, dtype=np.float32)
    x = pts[:,0]; y = pts[:,1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    try:
        c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy = c, d
        r = math.sqrt(max(e + cx**2 + cy**2, 0))
        return (cx, cy), r
    except Exception:
        return None, None


def redness_index(bgr):
    # Simple per-pixel redness: R - (G+B)/2 normalized to [0,1] via sigmoid-ish mapping
    b,g,r = cv2.split(bgr)
    ri = r.astype(np.float32) - (g.astype(np.float32) + b.astype(np.float32))/2.0
    ri = 1/(1+np.exp(-ri/20.0))  # squashed; empiric
    return ri


def mean_intensity(gray, mask):
    m = cv2.mean(gray, mask=mask)[0]
    return m if not np.isnan(m) else 0.0


def extract_eye_metrics(img_bgr, landmarks):
    h, w = img_bgr.shape[:2]
    # Convert landmarks (normalized) to pixel coords
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    # Build eye polygons and iris sets
    left_eye_pts = [pts[i] for i in LEFT_EYE_LANDMARKS]
    right_eye_pts = [pts[i] for i in RIGHT_EYE_LANDMARKS]
    left_iris_pts = [pts[i] for i in LEFT_IRIS_LMS]
    right_iris_pts = [pts[i] for i in RIGHT_IRIS_LMS]

    left_eye_mask = polygon_mask(img_bgr.shape, left_eye_pts)
    right_eye_mask = polygon_mask(img_bgr.shape, right_eye_pts)

    # Tighten sclera mask by removing inner iris region (approx)
    # Create iris masks via circles
    (lcx, lcy), lr = circle_from_points(left_iris_pts)
    (rcx, rcy), rr = circle_from_points(right_iris_pts)

    sclera_left_mask = left_eye_mask.copy()
    sclera_right_mask = right_eye_mask.copy()
    if lcx is not None and lr is not None and lr > 0:
        cv2.circle(sclera_left_mask, (int(lcx), int(lcy)), int(lr*1.1), 0, -1)
    if rcx is not None and rr is not None and rr > 0:
        cv2.circle(sclera_right_mask, (int(rcx), int(rcy)), int(rr*1.1), 0, -1)

    # Redness proxy in sclera
    ri = redness_index(img_bgr)
    left_red = cv2.mean(ri, mask=sclera_left_mask)[0]
    right_red = cv2.mean(ri, mask=sclera_right_mask)[0]

    # Aperture (ptosis proxy): vertical distance between eyelids normalized by eye width
    # Use top/bottom points of eye polygon approx
    def vert_aperture(eye_pts):
        ys = [p[1] for p in eye_pts]
        xs = [p[0] for p in eye_pts]
        height = (max(ys) - min(ys)) + 1e-6
        width = (max(xs) - min(xs)) + 1e-6
        return height / width

    left_ap = vert_aperture(left_eye_pts)
    right_ap = vert_aperture(right_eye_pts)

    # Strabismus proxy: compare gaze vectors (iris center relative to eye box center)
    def gaze_vector(eye_pts, center):
        xs = [p[0] for p in eye_pts]; ys = [p[1] for p in eye_pts]
        ex = (min(xs) + max(xs)) / 2.0
        ey = (min(ys) + max(ys)) / 2.0
        return np.array([center[0]-ex, center[1]-ey], dtype=np.float32)

    left_center = (lcx, lcy) if lcx is not None else (np.mean([p[0] for p in left_eye_pts]), np.mean([p[1] for p in left_eye_pts]))
    right_center = (rcx, rcy) if rcx is not None else (np.mean([p[0] for p in right_eye_pts]), np.mean([p[1] for p in right_eye_pts]))

    gv_left = gaze_vector(left_eye_pts, left_center)
    gv_right = gaze_vector(right_eye_pts, right_center)

    # Anisocoria: iris radius diff ratio (pupil radius ~ iris radius in MP landmarks proxy; very rough)
    anis_ratio = 0.0
    if lr and rr and lr > 0 and rr > 0:
        anis_ratio = abs(lr - rr) / max(lr, rr)

    # Leukocoria check: unusually bright pupil region (white reflex) compared to iris ring
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    leuk_left = 0.0; leuk_right = 0.0
    if lcx and lr:
        pupil_mask = np.zeros_like(gray)
        cv2.circle(pupil_mask, (int(lcx), int(lcy)), int(lr*0.6), 255, -1)
        leuk_left = mean_intensity(gray, pupil_mask)
    if rcx and rr:
        pupil_mask = np.zeros_like(gray)
        cv2.circle(pupil_mask, (int(rcx), int(rcy)), int(rr*0.6), 255, -1)
        leuk_right = mean_intensity(gray, pupil_mask)

    # --- Pupil detection inside annotated white circle ---
    def detect_pupil_radius(img_bgr, center, radius):
        if center is None or radius <= 0:
            return 0.0
        x, y, r = int(center[0]), int(center[1]), int(radius)
        # Crop a square region around the circle
        pad = int(r * 1.1)
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + pad, w), min(y + pad, h)
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Mask for the white circle
        mask = np.zeros_like(gray_roi)
        cv2.circle(mask, (pad, pad), r, 255, -1)
        # Threshold to find dark pupil
        _, thresh = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.bitwise_and(thresh, mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pupil_r = 0.0
        for cnt in contours:
            (cx, cy), pr = cv2.minEnclosingCircle(cnt)
            if pr > pupil_r:
                pupil_r = pr
        return float(pupil_r)

    left_pupil_r = detect_pupil_radius(img_bgr, (lcx, lcy), lr*0.6)
    right_pupil_r = detect_pupil_radius(img_bgr, (rcx, rcy), rr*0.6)
    print(f"Detected left pupil radius: {left_pupil_r:.2f}, right pupil radius: {right_pupil_r:.2f}")

    # Compile metrics
    metrics = {
        "left_redness": float(left_red),
        "right_redness": float(right_red),
        "left_aperture": float(left_ap),
        "right_aperture": float(right_ap),
        "gaze_left": [float(gv_left[0]), float(gv_left[1])],
        "gaze_right": [float(gv_right[0]), float(gv_right[1])],
        "anisocoria_ratio": float(abs(left_pupil_r - right_pupil_r) / max(left_pupil_r, right_pupil_r) if left_pupil_r > 0 and right_pupil_r > 0 else 0.0),
        "leukocoria_left_brightness": float(leuk_left),
        "leukocoria_right_brightness": float(leuk_right),
        "left_iris_radius_px": float(lr if lr else 0.0),
        "right_iris_radius_px": float(rr if rr else 0.0),
        "left_pupil_radius_px": float(left_pupil_r),
        "right_pupil_radius_px": float(right_pupil_r),
    }
    # Also return polygon points for drawing
    geometry = {
        "left_eye_pts": left_eye_pts,
        "right_eye_pts": right_eye_pts,
        "left_iris_center": (int(left_center[0]), int(left_center[1])) if left_center else None,
        "right_iris_center": (int(right_center[0]), int(right_center[1])) if right_center else None,
        "left_iris_r": int(lr) if lr else 0,
        "right_iris_r": int(rr) if rr else 0,
        "left_pupil_center": (int(lcx), int(lcy)) if lcx and lcy else None,
        "right_pupil_center": (int(rcx), int(rcy)) if rcx and rcy else None,
        "left_pupil_r": int(left_pupil_r),
        "right_pupil_r": int(right_pupil_r),
    }
    return metrics, geometry


def evaluate_findings(metrics):
    findings = []

    # Heuristic thresholds (empiric; adjust with data!)
    # Redness: ri ~ 0.5 baseline, >0.6-0.65 suggests redness
    left_red = metrics["left_redness"]
    right_red = metrics["right_redness"]
    red_score = max(left_red, right_red)
    red_present = red_score > 0.63
    findings.append(Finding(
        name="Scleral redness (conjunctivitis/irritation proxy)",
        present=bool(red_present),
        confidence=float(min(max((red_score-0.55)/0.25, 0), 1)),
        reasoning=f"Redness index L={left_red:.2f}, R={right_red:.2f}. Threshold ~0.63."
    ))

    # Ptosis: aperture asymmetry > 20% or absolute aperture < 0.18 (tight lids) (very rough, camera-angle sensitive)
    la = metrics["left_aperture"]; ra = metrics["right_aperture"]
    asym = abs(la - ra) / max(la, ra)
    ptosis_present = (asym > 0.20) or (min(la, ra) < 0.18)
    findings.append(Finding(
        name="Ptosis (eyelid droop proxy via aperture)",
        present=bool(ptosis_present),
        confidence=float(min(max((asym-0.10)/0.25, 0), 1)),
        reasoning=f"Aperture L={la:.2f}, R={ra:.2f}, asymmetry={asym:.2f}. Angle/pose can confound."
    ))

    # Strabismus: gaze vector angle/length mismatch between eyes (normalized by eye size).
    gl = np.array(metrics["gaze_left"], dtype=np.float32)
    gr = np.array(metrics["gaze_right"], dtype=np.float32)
    # Compare angles (in degrees)
    def angle(v):
        return math.degrees(math.atan2(float(v[1]), float(v[0]) + 1e-6))
    ang_diff = abs(angle(gl) - angle(gr))
    ang_diff = min(ang_diff, 360-ang_diff)
    strab_present = ang_diff > 8.0  # rough
    findings.append(Finding(
        name="Strabismus screening (gaze asymmetry)",
        present=bool(strab_present),
        confidence=float(min(max((ang_diff-5)/20, 0), 1)),
        reasoning=f"Gaze angle diff ≈ {ang_diff:.1f}°. Requires frontal, well-lit image."
    ))

    # Anisocoria: iris radius (pupil proxy) difference > 20%
    anis = metrics["anisocoria_ratio"]
    ani_present = anis > 0.20
    findings.append(Finding(
        name="Anisocoria (pupil size difference)",
        present=bool(ani_present),
        confidence=float(min(max((anis-0.10)/0.30, 0), 1)),
        reasoning=f"Relative radius difference ≈ {anis*100:.1f}%."
    ))

    # Leukocoria (white reflex) – conservative: unusually high brightness vs overall eye average
    # We'll compute a z-like score by rough normalization later; here just threshold brightness
    leuk_b = max(metrics["leukocoria_left_brightness"], metrics["leukocoria_right_brightness"])
    leuk_present = leuk_b > 200  # bright on 0-255 scale (depends on exposure/flash)
    findings.append(Finding(
        name="Leukocoria (white pupillary reflex) – cancer risk flag",
        present=bool(leuk_present),
        confidence=float(0.6 if leuk_present else 0.0),
        reasoning=f"Bright pupil intensity ≈ {leuk_b:.1f}/255; ONLY a rough flag—seek urgent clinical exam if seen repeatedly."
    ))

    return findings


def draw_annotations(img_bgr, geometry, metrics, findings):
    out = img_bgr.copy()
    # Draw eye polygons
    for eye_pts in [geometry["left_eye_pts"], geometry["right_eye_pts"]]:
        cv2.polylines(out, [np.array(eye_pts, dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
    # Draw iris circles
    if geometry["left_iris_center"] and geometry["left_iris_r"] > 0:
        cv2.circle(out, geometry["left_iris_center"], geometry["left_iris_r"], (255,0,0), 1, cv2.LINE_AA)
    if geometry["right_iris_center"] and geometry["right_iris_r"] > 0:
        cv2.circle(out, geometry["right_iris_center"], geometry["right_iris_r"], (255,0,0), 1, cv2.LINE_AA)
    # Draw pupil circles (white)
    if geometry["left_pupil_center"] and geometry["left_pupil_r"] > 0:
        cv2.circle(out, geometry["left_pupil_center"], geometry["left_pupil_r"], (255,255,255), 2, cv2.LINE_AA)
    if geometry["right_pupil_center"] and geometry["right_pupil_r"] > 0:
        cv2.circle(out, geometry["right_pupil_center"], geometry["right_pupil_r"], (255,255,255), 2, cv2.LINE_AA)
    # Overlay findings text
    y = 20
    for f in findings:
        text = f"{f.name}: {'FLAG' if f.present else 'ok'} ({f.confidence:.2f})"
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255) if f.present else (200,200,200), 1, cv2.LINE_AA)
        y += 18
    return out


def run_on_image(image_path, out_prefix):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise SystemExit(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]
    print(f"Loaded image: {image_path} ({w}x{h} pixels)")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as fm:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            print("FaceMesh failed. Trying OpenCV Haar eye detector fallback...")
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(eyes) < 2:
                debug_path = f"{out_prefix}_debug.jpg"
                cv2.imwrite(debug_path, img_bgr)
                print(f"No face or eyes detected.\n- Saved debug copy: {debug_path}\n- Tips: Use a sharp, frontal, well-lit image. Avoid sunglasses, masks, or extreme angles.")
                raise SystemExit("Detection failed. See above for details.")
            # Take the two largest detected eyes
            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            metrics = {}
            geometry = {}
            for idx, (ex, ey, ew, eh) in enumerate(eyes):
                eye_roi = img_bgr[ey:ey+eh, ex:ex+ew]
                center = (ex + ew//2, ey + eh//2)
                r = min(ew, eh)//2
                pupil_r = 0.0
                # Use same pupil detection as before
                def detect_pupil_radius_eye(roi, r):
                    if roi.size == 0:
                        return 0.0
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(gray_roi)
                    cv2.circle(mask, (roi.shape[1]//2, roi.shape[0]//2), r, 255, -1)
                    _, thresh = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
                    thresh = cv2.bitwise_and(thresh, mask)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pupil_r = 0.0
                    for cnt in contours:
                        (_, _), pr = cv2.minEnclosingCircle(cnt)
                        if pr > pupil_r:
                            pupil_r = pr
                    return float(pupil_r)
                pupil_r = detect_pupil_radius_eye(eye_roi, int(r*0.6))
                metrics[f"eye{idx}_pupil_radius_px"] = pupil_r
                geometry[f"eye{idx}_center"] = center
                geometry[f"eye{idx}_r"] = int(r)
            # Calculate anisocoria
            left_pupil_r = metrics.get("eye0_pupil_radius_px", 0.0)
            right_pupil_r = metrics.get("eye1_pupil_radius_px", 0.0)
            metrics["anisocoria_ratio"] = float(abs(left_pupil_r - right_pupil_r) / max(left_pupil_r, right_pupil_r) if left_pupil_r > 0 and right_pupil_r > 0 else 0.0)
            print(f"Fallback detected left pupil radius: {left_pupil_r:.2f}, right pupil radius: {right_pupil_r:.2f}")
            # Draw fallback annotations
            annotated = img_bgr.copy()
            for idx in range(2):
                center = geometry.get(f"eye{idx}_center")
                r = geometry.get(f"eye{idx}_r")
                pupil_r = int(metrics.get(f"eye{idx}_pupil_radius_px", 0))
                if center and r > 0:
                    cv2.circle(annotated, center, r, (255,0,0), 1, cv2.LINE_AA)  # iris proxy
                if center and pupil_r > 0:
                    cv2.circle(annotated, center, pupil_r, (255,255,255), 2, cv2.LINE_AA)  # pupil
            annotated_path = f"{out_prefix}_annotated.jpg"
            cv2.imwrite(annotated_path, annotated)
            print(f"Fallback annotated image saved: {annotated_path}")
            # Minimal findings for fallback
            findings = []
            anis = metrics["anisocoria_ratio"]
            ani_present = anis > 0.20
            findings.append(Finding(
                name="Anisocoria (pupil size difference)",
                present=bool(ani_present),
                confidence=float(min(max((anis-0.10)/0.30, 0), 1)),
                reasoning=f"Relative radius difference ≈ {anis*100:.1f}% (fallback)."
            ))
            # Save minimal report
            rep = Report(
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                image_path=image_path,
                notes="Fallback: Only eyes detected. Prototype eye-screening from cropped eye image.",
                findings=[asdict(f) for f in findings]
            )
            json_path = f"{out_prefix}_report.json"
            txt_path = f"{out_prefix}_report.txt"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(rep), f, indent=2)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Report time: {rep.timestamp}\nImage: {rep.image_path}\nNotes: {rep.notes}\n\n")
                for fnd in findings:
                    f.write(f"- {fnd.name}\n  Present: {fnd.present}\n  Confidence: {fnd.confidence:.2f}\n  Reasoning: {fnd.reasoning}\n\n")
            return metrics, geometry

        lms = res.multi_face_landmarks[0].landmark
        metrics, geometry = extract_eye_metrics(img_bgr, lms)
        findings = evaluate_findings(metrics)

        annotated = draw_annotations(img_bgr, geometry, metrics, findings)
        annotated_path = f"{out_prefix}_annotated.jpg"
        cv2.imwrite(annotated_path, annotated)

        rep = Report(
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            image_path=image_path,
            notes=("Prototype eye-screening from single image. NOT a diagnosis. "
                   "Ensure frontal pose, no glasses (if possible), good light."),
            findings=[asdict(f) for f in findings]
        )
        json_path = f"{out_prefix}_report.json"
        txt_path = f"{out_prefix}_report.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(rep), f, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Report time: {rep.timestamp}\nImage: {rep.image_path}\nNotes: {rep.notes}\n\n")
            for fnd in findings:
                f.write(f"- {fnd.name}\n  Present: {fnd.present}\n  Confidence: {fnd.confidence:.2f}\n  Reasoning: {fnd.reasoning}\n\n")

        return {
            "annotated_image": annotated_path,
            "json_report": json_path,
            "text_report": txt_path,
            "metrics": metrics
        }


def capture_webcam_and_save(path="webcam_capture.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam.")
    print("Press SPACE to capture, ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            path = None
            break
        if key == 32:  # SPACE
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")
            break
    cap.release()
    cv2.destroyAllWindows()
    return path


def main():
    parser = argparse.ArgumentParser(description="Prototype eye-condition screening (single image). NOT a medical device.")
    parser.add_argument("--image", type=str, help="Path to an input face image (frontal).")
    parser.add_argument("--capture", action="store_true", help="Capture a frame from webcam instead of using --image.")
    parser.add_argument("--out", type=str, default="eye_screening_output", help="Output prefix path.")
    args = parser.parse_args()

    if args.capture:
        saved = capture_webcam_and_save("webcam_capture.jpg")
        if saved is None:
            print("Cancelled.")
            return
        image_path = saved
    elif args.image:
        image_path = args.image
    else:
        raise SystemExit("Provide --image path or use --capture.")

    results = run_on_image(image_path, args.out)
    print("Done.")
    print("Annotated image:", results["annotated_image"])
    print("JSON report:", results["json_report"])
    print("Text report:", results["text_report"])


if __name__ == "__main__":
    main()