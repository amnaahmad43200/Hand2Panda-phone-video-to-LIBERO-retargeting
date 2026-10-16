import argparse
import json
import os

import cv2
import numpy as np
from scipy.signal import savgol_filter

CORNER_IDS = (0, 1, 2, 3)
OBJ_ID, PLATE_ID = 10, 11
MOVE_THRESH = 0.015          # m, motion onset / offset threshold
MAX_GAP_S = 0.4              # longest gap we are willing to interpolate


def detect_all(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    frames = []                                   # list of {id: (4,2) corners}
    first_frame = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if first_frame is None:
            first_frame = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        d = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                d[int(i)] = c.reshape(4, 2).astype(np.float64)
        frames.append(d)
    cap.release()
    return frames, fps, (W, H), first_frame


def side_px(c):
    return float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))


def centre(c):
    return c.mean(axis=0)


def calibrate_table(frames, table_w, table_h, corner_mm):
    """Homography image->table (metres) plus nadir and table-marker pixel size."""
    table_xy = np.array([[0, 0], [table_w, 0], [table_w, table_h], [0, table_h]], np.float64)
    full = [f for f in frames if all(i in f for i in CORNER_IDS)]
    if len(full) < 10:
        raise RuntimeError(f"corner markers 0-3 all visible in only {len(full)} frames; "
                           "check that all four are in view and not occluded at the start")
    full = full[: max(10, min(len(full), 90))]
    ctr = np.array([[np.median([centre(f[i])[k] for f in full]) for k in (0, 1)] for i in CORNER_IDS])
    s_tab = float(np.median([side_px(f[i]) for f in full for i in CORNER_IDS]))
    Hm, _ = cv2.findHomography(ctr, table_xy)
    return Hm, s_tab, ctr


def to_table(Hm, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, Hm).reshape(-1, 2)


def marker_series(frames, mid, Hm, nadir, s_tab, size_ratio, z0):
    """Per-frame (x, y, h) of one marker in the table frame; NaN where not detected.
    size_ratio = true_marker_size / true_table_marker_size."""
    T = len(frames)
    out = np.full((T, 3), np.nan)
    for t, f in enumerate(frames):
        if mid not in f:
            continue
        p_app = to_table(Hm, centre(f[mid]))[0]
        r = size_ratio * s_tab / side_px(f[mid])        # Z / Z0
        out[t, :2] = nadir + (p_app - nadir) * r
        out[t, 2] = z0 * (1.0 - r)
    return out


def fill_gaps(x, fps):
    """Linearly interpolate NaN gaps; return (filled, longest_gap_seconds, detection_rate)."""
    x = x.copy()
    bad = np.isnan(x[:, 0])
    if bad.all():
        raise RuntimeError("marker never detected")
    idx = np.arange(len(x))
    longest, run = 0, 0
    for b in bad:
        run = run + 1 if b else 0
        longest = max(longest, run)
    for k in range(x.shape[1]):
        x[bad, k] = np.interp(idx[bad], idx[~bad], x[~bad, k])
    return x, longest / fps, 1.0 - bad.mean()


def smooth(x, fps):
    win = int(round(0.2 * fps)) | 1                      # ~0.2 s, odd
    win = max(win, 5)
    if len(x) <= win:
        return x
    return savgol_filter(x, win, 2, axis=0)


def segment(xyz, fps):
    """Return (grasp_idx, release_idx, start_xyz, end_xyz) from the object's own motion."""
    n0 = max(3, int(0.5 * fps))
    start = np.median(xyz[:n0], axis=0)
    end = np.median(xyz[-n0:], axis=0)
    moving_from_start = np.linalg.norm(xyz - start, axis=1) > MOVE_THRESH
    moving_from_end = np.linalg.norm(xyz - end, axis=1) > MOVE_THRESH
    if not moving_from_start.any():
        raise RuntimeError("object never moved")
    g = int(np.argmax(moving_from_start))
    r = int(len(xyz) - 1 - np.argmax(moving_from_end[::-1]))
    if r <= g + 3:
        raise RuntimeError(f"degenerate transit window ({g}, {r})")
    return g, r, start, end


def extract(video, table_w=0.70, table_h=0.36, cam_height=0.65, corner_mm=60, obj_mm=40,
            plate_mm=40, layout="", debug_png=None):
    frames, fps, (W, H), first = detect_all(video)
    Hm, s_tab, _ = calibrate_table(frames, table_w, table_h, corner_mm)
    nadir = to_table(Hm, [[W / 2, H / 2]])[0]
    obj_raw = marker_series(frames, OBJ_ID, Hm, nadir, s_tab, obj_mm / corner_mm, cam_height)
    plate_raw = marker_series(frames, PLATE_ID, Hm, nadir, s_tab, plate_mm / corner_mm, cam_height)
    obj, longest_gap, det_rate = fill_gaps(obj_raw, fps)
    obj = smooth(obj, fps)
    g, r, start, end = segment(obj, fps)
    h_rest = float(np.median(obj[: max(3, int(0.5 * fps)), 2]))
    xyz = obj.copy()
    xyz[:, 2] -= h_rest                                   # height above the object's resting height
    plate_ok = ~np.isnan(plate_raw[:, 0])
    plate_xy = np.median(plate_raw[plate_ok, :2], axis=0) if plate_ok.any() else np.full(2, np.nan)

    qc = {
        "video": os.path.basename(video), "layout": layout, "fps": fps, "n_frames": len(frames),
        "obj_detect_rate": round(det_rate, 3), "longest_gap_s": round(longest_gap, 3),
        "grasp_idx": g, "release_idx": r, "transit_s": round((r - g) / fps, 3),
        "rest_height_est_m": round(h_rest, 4),
        "max_lift_m": round(float(xyz[g:r + 1, 2].max()), 4),
        "start_to_end_m": round(float(np.linalg.norm(end[:2] - start[:2])), 4),
        "end_to_plate_marker_m": round(float(np.linalg.norm(end[:2] - plate_xy)), 4) if plate_ok.any() else None,
    }
    warnings = []
    if longest_gap > MAX_GAP_S:
        warnings.append(f"longest detection gap {longest_gap:.2f}s > {MAX_GAP_S}s: path may be unreliable")
    if det_rate < 0.7:
        warnings.append(f"object marker detected in only {det_rate:.0%} of frames")
    if qc["end_to_plate_marker_m"] is not None and qc["end_to_plate_marker_m"] > 0.06:
        warnings.append("object ended >6 cm from plate marker: check marker placement / that it was placed on the plate")
    if not (0.0 <= qc["max_lift_m"] <= 0.25):
        warnings.append(f"implausible lift {qc['max_lift_m']} m: check --cam-height")
    qc["warnings"] = warnings
    qc["ok"] = not warnings

    result = dict(t=np.arange(len(frames)) / fps, xyz=xyz, xyz_raw=obj_raw, start_xyz=start, end_xyz=end,
                  plate_xy=plate_xy, grasp_idx=g, release_idx=r, fps=fps)
    if debug_png:
        save_debug(debug_png, result, qc, table_w, table_h)
    return result, qc


def save_debug(path, res, qc, table_w, table_h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz, g, r = res["xyz"], res["grasp_idx"], res["release_idx"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(xyz[:, 0], xyz[:, 1], "-", c="0.7", lw=1)
    ax[0].plot(xyz[g:r + 1, 0], xyz[g:r + 1, 1], "-", c="C0", lw=2, label="transit")
    ax[0].plot(*res["start_xyz"][:2], "go", label="start")
    ax[0].plot(*res["end_xyz"][:2], "rs", label="end")
    if not np.isnan(res["plate_xy"]).any():
        ax[0].plot(*res["plate_xy"], "k*", ms=12, label="plate marker")
    ax[0].set_xlim(0, table_w); ax[0].set_ylim(0, table_h); ax[0].set_aspect("equal")
    ax[0].set_title("top-down (table frame, m)"); ax[0].legend(fontsize=7)
    ax[1].plot(res["t"], xyz[:, 2] * 100)
    ax[1].axvspan(res["t"][g], res["t"][r], color="C0", alpha=0.15)
    ax[1].set_xlabel("s"); ax[1].set_ylabel("lift above rest (cm)"); ax[1].set_title("height from marker scale")
    fig.suptitle(f"{qc['video']}  ok={qc['ok']}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="output .npz (a .json QC file and _qc.png are written beside it)")
    ap.add_argument("--layout", default="")
    ap.add_argument("--table-w", type=float, default=0.70, help="distance between corner markers 0 and 1 (m)")
    ap.add_argument("--table-h", type=float, default=0.36, help="distance between corner markers 0 and 3 (m)")
    ap.add_argument("--cam-height", type=float, default=0.65, help="lens-to-table distance (m), measured")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base = os.path.splitext(args.out)[0]
    res, qc = extract(args.video, args.table_w, args.table_h, args.cam_height, layout=args.layout,
                      debug_png=base + "_qc.png")
    np.savez(args.out, **res)
    json.dump(qc, open(base + ".json", "w"), indent=1)
    print(json.dumps(qc, indent=1))


if __name__ == "__main__":
    main()