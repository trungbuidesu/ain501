# Khối Xử Lý Cấu Trúc Sinh Học: MediaPipe

> [!NOTE] Bản Nháp Tài Liệu (Draft Note)
> Trang tài liệu này đang trong tiến trình soạn thảo (Work in Progress).

Đặc tả mốc neo kiến trúc `MediaPipe` siêu nhẹ (~1M tham số, khoảng 3MB lưu trữ). Giải pháp được thiết kế độc quyền nhằm rà soát và định vị lưới cấu trúc khuôn mặt dạng lưới (Face mesh) cùng các hình đồ phân rã khung xương thực thể (Pose estimation) theo thời gian thực. Sẽ cập nhật bảng tọa độ các khớp sớm.

## Week 5 verification (CPU, no fine-tuning)

**CLI:**

```text
python -m src.training.scripts.verify_prebuilt_week5 mediapipe --synthetic --limit 20
python -m src.training.scripts.verify_prebuilt_week5 mediapipe --image-dir path/to/faces --limit 20
```

Add `--full-landmarks` to embed all 478 face points in JSONL (large files). Default omits the point list but keeps counts and heuristics.

**Implementation:** MediaPipe Python 0.10+ uses the **Tasks** API (`FaceLandmarker`, `PoseLandmarker`). The verify script downloads `face_landmarker.task` and `pose_landmarker_lite.task` into `reports/prebuilt_week5/_mediapipe_models/` (needs network once).

**Outputs:**

- `reports/prebuilt_week5/mediapipe_face_pose_samples.jsonl`
- `reports/prebuilt_week5/mediapipe_benchmark_cpu.json`

**Coordinates:** Face mesh and pose landmarks use **normalized image coordinates** (`x`,`y` in [0,1], origin top-left; `z` is relative depth). Pose entries include per-joint `visibility` in [0,1].

**Face mesh:** up to 478 points per face (face landmarker task). Emotion-related output in reports is a **geometry heuristic** (`mouth_open_ratio`, `mouth_width_ratio`, `heuristic_label`) — not a FER-trained classifier.

**Heuristic landmark indices (documentation):** mouth vertical uses 13 and 14; mouth width uses 61 and 291; inter-eye span uses 33 and 263 (see [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)).

**Pose:** 33 BlazePose landmarks. Rule-based labels in `pose_rules.rule_label`:

| Label | Rule (summary) |
| --- | --- |
| `arms_up` | Both wrists above shoulders with visibility > 0.5 |
| `sit_like` | Torso vertical span (hip_y - shoulder_y) < 0.08 |
| `stand_like` | Default otherwise |

**JSONL schema (per image):** `face_ms`, `pose_ms`, `face` (detection, `landmark_count`, `emotion_proxy`), `pose` (`keypoints` array, `pose_rules`).