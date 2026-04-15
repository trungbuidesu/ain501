# Accessibility Vision Assistant — Implementation Prompt (Phase 1-4)

## SYSTEM RULES — ĐỌC KỸ TRƯỚC KHI LÀM BẤT CỨ GÌ

### Quy tắc tuyệt đối

1. **KHÔNG HỎI.** Prompt này đủ chi tiết. Nếu thiếu thông tin → đọc lại code/config/docs trong repo. Nếu vẫn không rõ → chọn option hợp lý nhất, ghi lại quyết định trong commit message. KHÔNG dừng lại hỏi "bạn muốn A hay B?"

2. **ĐỌC TRƯỚC, CODE SAU.** Trước khi viết bất kỳ dòng code nào, scan toàn bộ:
   - `src/` — tất cả modules hiện có, hiểu structure, imports, interfaces
   - `configs/` — tất cả YAML configs hiện có
   - `models/` — list tất cả ONNX files thực tế có trên disk
   - `docs/` — đọc architecture docs, training logs, checklists
   - `tests/` — hiểu test conventions, fixtures
   - `data/knowledge/` — nếu đã có JSON files
   Ghi ra 1 file `SCAN_REPORT.md` tạm (không commit) tóm tắt: modules nào đã tồn tại, interfaces nào đã define, ONNX files nào có, config keys nào đã dùng. Dùng file này làm reference khi implement.

3. **KHÔNG DUPLICATE.** Trước khi tạo function/class mới:
   - Grep toàn bộ `src/` xem logic đó đã tồn tại chưa
   - Nếu đã có → import và dùng lại, hoặc extend
   - Nếu tương tự → refactor module cũ thay vì tạo mới
   - KHÔNG BAO GIỜ có 2 functions làm cùng 1 việc trong repo

4. **SELF-TEST TRƯỚC KHI BÁO.** Sau mỗi phase, tự chạy và verify TRƯỚC khi report:
   ```
   black .
   ruff check . --fix
   mypy .
   pytest -q
   python -m src.app --config configs/app.yaml --dry-run  # nếu có
   ```
   Nếu fail → fix → chạy lại → chỉ report khi TẤT CẢ PASS.

5. **CODE QUALITY CHECK cuối mỗi phase:**
   - Scan unused imports: `ruff check . --select F401`
   - Scan dead code: tìm functions/classes không được import/gọi từ đâu
   - Scan hardcoded paths: grep `"models/`, `"data/`, `"configs/` → phải dùng config
   - Scan circular imports: verify mỗi module import graph không có cycle
   - Verify tất cả ONNX paths trong code khớp với files thực tế trong `models/`
   - Verify tất cả config keys trong code khớp với `configs/app.yaml`

6. **KHÔNG SỬA `models/` hoặc `training/`.** Chỉ consume ONNX files. Không retrain, không modify training scripts.

7. **COMMIT MESSAGE FORMAT:** `phase-{N}.{sub}: {mô tả ngắn}`. Ví dụ: `phase-1.2: implement object detection agent with ONNX Runtime`

8. **CHUẨN HÓA THƯ MỤC & FILE — BẮT BUỘC.** Đây là cấu trúc duy nhất được phép. KHÔNG tạo file/folder ngoài structure này. KHÔNG đặt tên file theo phase/batch/week.

   **Cấu trúc bắt buộc:**
   ```
   project/
   ├── src/
   │   ├── __init__.py
   │   ├── app.py                          # Entry point duy nhất
   │   ├── capture/
   │   │   ├── __init__.py
   │   │   ├── base.py                     # InputSource ABC
   │   │   ├── webcam.py                   # WebcamSource
   │   │   ├── screen.py                   # ScreenSource
   │   │   └── file.py                     # FileSource
   │   ├── agents/
   │   │   ├── __init__.py
   │   │   ├── base.py                     # Agent ABC
   │   │   ├── object_agent.py             # YOLOv8n ONNX
   │   │   ├── action_agent.py             # MoViNet ONNX
   │   │   ├── ocr_agent.py                # PaddleOCR
   │   │   ├── face_pose_agent.py          # MediaPipe
   │   │   └── agent_manager.py            # ThreadPoolExecutor orchestration
   │   ├── router/
   │   │   ├── __init__.py
   │   │   └── scene_router.py             # ONNX hoặc rule-based
   │   ├── caption/
   │   │   ├── __init__.py
   │   │   ├── template_engine.py          # Simple templates (fallback)
   │   │   └── orchestrator.py             # RAG-enhanced (primary)
   │   ├── rag/
   │   │   ├── __init__.py
   │   │   └── knowledge_base.py           # FAISS + MiniLM ONNX
   │   ├── output/
   │   │   ├── __init__.py
   │   │   └── tts_engine.py               # Piper / pyttsx3
   │   └── core/
   │       ├── __init__.py
   │       ├── pipeline.py                 # Threading: CaptureThread, ProcessingThread
   │       ├── config.py                   # Config loader
   │       └── types.py                    # Shared dataclasses: Detection, Action, TextRegion, etc.
   ├── configs/
   │   └── app.yaml                        # TẤT CẢ app config tập trung 1 file
   ├── data/
   │   └── knowledge/
   │       ├── hazard_patterns.json
   │       ├── spatial_templates.json
   │       └── relationship_patterns.json
   ├── models/                             # ONNX files — KHÔNG SỬA, chỉ đọc
   ├── tests/
   │   ├── test_capture.py
   │   ├── test_agents.py
   │   ├── test_router.py
   │   ├── test_caption.py
   │   ├── test_rag.py
   │   ├── test_tts.py
   │   └── test_pipeline.py
   └── docs/
       ├── architecture.md
       ├── setup.md
       └── README.md
   ```

   **Quy tắc đặt tên:**
   - File name: `snake_case.py`, KHÔNG có prefix phase/week/batch (❌ `phase1_detector.py`, ❌ `week6_benchmark.py`, ❌ `v2_template.py`)
   - Class name: `PascalCase` (✅ `ObjectAgent`, ❌ `Object_Agent`, ❌ `objectagent`)
   - 1 module = 1 responsibility. KHÔNG tạo file `utils.py` hoặc `helpers.py` chứa đủ thứ
   - Shared types/dataclasses tập trung trong `src/core/types.py` — KHÔNG define cùng dataclass ở 2 files khác nhau
   - Config tập trung trong `configs/app.yaml` — KHÔNG tạo `config_phase1.yaml`, `config_v2.yaml`

   **Trước khi tạo file mới, kiểm tra:**
   - File này thuộc folder nào trong structure trên?
   - Nếu không thuộc folder nào → KHÔNG TẠO. Đặt logic vào module phù hợp nhất
   - Nếu cần folder mới → ghi lý do trong commit message, phải là folder có ý nghĩa (`src/audio/` OK, `src/new_stuff/` KHÔNG OK)

   **Dọn dẹp file cũ:**
   - Scan repo tìm files nằm ngoài structure: scripts lẻ, notebooks rác, file tạm đặt tên theo phase
   - Di chuyển vào đúng vị trí hoặc xóa nếu không cần
   - KHÔNG giữ file cũ "phòng khi cần" — git đã lưu history

---

## Implementation Plan

Implement tuần tự Phase 1 → 2 → 3 → 4. Mỗi phase phải pass checkpoint trước khi sang phase tiếp.

---

## Phase 1 — MVP: Capture + Detection + TTS

### 1.1 Input module (`src/capture/`)

- Abstract base class `InputSource` với `read_frame() -> Optional[np.ndarray]`
- 3 implementations: `WebcamSource(device_id)`, `ScreenSource(region)`, `FileSource(path)`
- Config: `configs/app.yaml` → `input.mode: webcam | screen | file`
- Ring buffer giữ 10 frames (dùng `collections.deque(maxlen=10)`)
- FPS control: `time.sleep()` hoặc frame skip, default 2 FPS từ config
- Change detection: SSIM diff (`skimage.metrics.structural_similarity`), threshold từ config, skip nếu không đổi
- **Check trước:** `src/` có module capture nào chưa? Nếu có → extend, không tạo mới

### 1.2 Object detection agent (`src/agents/object_agent.py`)

- Load ONNX qua `onnxruntime.InferenceSession`
- **Check trước:** verify file `models/yolov8n.onnx` hoặc `models/yolov8n_int8.onnx` tồn tại. Dùng file nào có. Nếu không có ONNX → check có `yolov8n.pt` không → export ONNX trước
- Preprocessing: resize 640x640, normalize, HWC→CHW, add batch dim
- Postprocessing: parse YOLO output format, NMS, confidence filter
- Output: `List[Detection]` — mỗi Detection có: `label: str, confidence: float, bbox: Tuple[x1,y1,x2,y2], position: str, distance: str`
- Position logic: bbox center x < 0.33 → "bên trái", > 0.66 → "bên phải", else → "phía trước"
- Distance logic: bbox area / frame area > 0.15 → "gần", < 0.03 → "xa", else → ""
- Class mapping: COCO class IDs → accessibility labels (person=0, bicycle=1, car=2, dog=16, ...)
- **Check trước:** `src/agents/` có object detection code nào chưa? Grep "yolo", "detection", "onnxruntime"

### 1.3 Spatial template engine (`src/caption/template_engine.py`)

- Input: `List[Detection]` → Output: `str` (caption tiếng Việt)
- Group detections by label, count duplicates: "2 người phía trước"
- Priority sort trước khi generate:
  - P0 hazard: car, truck, bus, motorcycle, bicycle (moving vehicles)
  - P1 navigation: stairs, traffic_light, stop_sign, door (nếu có)
  - P2 info: person, dog, bench, chair, backpack
- Dedup: cache previous caption, skip nếu similarity > 80% (simple string match)
- **Check trước:** `src/caption/` có template code nào chưa?

### 1.4 TTS module (`src/output/tts_engine.py`)

- Try import `piper` → dùng Piper TTS offline
- Fallback: `pyttsx3` (system TTS)
- Fallback 2: `print()` to console nếu cả hai fail (dev mode)
- Thread-safe: TTS chạy trong `threading.Thread(daemon=True)`
- Priority queue: `queue.PriorityQueue` — hazard=0, navigation=1, info=2
- Interrupt: nhận P0 message → clear queue, đọc P0 ngay
- Config: voice, speed, volume từ `configs/app.yaml` → `tts.*`
- **Check trước:** `src/output/` hoặc `src/tts/` có TTS code nào chưa?

### 1.5 Main loop (`src/app.py`)

```python
def main(config_path: str = "configs/app.yaml"):
    config = load_config(config_path)
    capture = InputSource.from_config(config)
    detector = ObjectAgent.from_config(config)
    template = TemplateEngine()
    tts = TTSEngine.from_config(config)

    tts.speak("Trợ lý đã sẵn sàng", priority=1)

    try:
        while True:
            frame = capture.read_frame()
            if frame is None:
                continue
            if not capture.has_changed(frame):
                continue
            detections = detector.detect(frame)
            if not detections:
                continue
            caption = template.generate(detections)
            if caption:
                tts.speak(caption, priority=2)
    except KeyboardInterrupt:
        tts.speak("Đã tắt", priority=1)
        capture.release()
```

- Entry point: `python -m src.app` hoặc `python -m src.app --config configs/app.yaml`
- `--dry-run` flag: chạy 10 frames rồi exit, in captions ra console, không TTS

### Checkpoint Phase 1

Tự chạy verify:
- `python -m src.app --dry-run` → phải in detections + captions ra console
- `python -m src.app` với webcam → phải nghe TTS đọc objects
- `pytest -q` pass
- Không có unused imports, dead code
- Report: list files created/modified, test results

---

## Phase 2 — Router + Multi-Agent

### 2.1 Router (`src/router/scene_router.py`)

- **Check trước:** `models/` có `scene_classifier.onnx` không?
  - Nếu CÓ: load ONNX, classify frame → scene type
  - Nếu KHÔNG: dùng rule-based fallback:
    - Frame diff > threshold → `motion`
    - Canny edge density > threshold → `text_present`
    - Skin color ratio > threshold → `face_detected`
    - Else → `static`
- Output: `Set[str]` — có thể multiple: `{"motion", "text_present"}`
- Routing table (hardcode OK, không cần RAG cho routing):
  ```python
  ROUTE = {
      "motion": ["object_agent", "action_agent"],
      "text_present": ["object_agent", "ocr_agent"],
      "face_detected": ["object_agent", "face_agent"],
      "static": ["object_agent"],
      "hazard": ["object_agent", "action_agent"],
  }
  ```

### 2.2 Action agent (`src/agents/action_agent.py`)

- **Check trước:** verify `models/movinet_a0_int8.onnx` tồn tại
- Input: 16 frames từ ring buffer, stride 2
- Output: `Action(label: str, confidence: float)` — walk, run, wave, sit, stand, climb_stairs, etc.
- Only trigger khi router returns `motion`

### 2.3 OCR agent (`src/agents/ocr_agent.py`)

- `from paddleocr import PaddleOCR` — lazy init, chỉ load khi cần
- Output: `List[TextRegion(text: str, bbox, confidence: float)]`
- Filter: drop text với confidence < 0.5
- Only trigger khi router returns `text_present`

### 2.4 Face/Pose agent (`src/agents/face_pose_agent.py`)

- `import mediapipe as mp` — lazy init
- Output: face count, dominant emotion proxy (smile score), pose type
- Only trigger khi router returns `face_detected`

### 2.5 Agent manager (`src/agents/agent_manager.py`)

- `concurrent.futures.ThreadPoolExecutor(max_workers=4)`
- Input: frame + scene_types from router
- Logic: chỉ submit agents mà routing table chỉ định
- Timeout: `future.result(timeout=0.2)` — 200ms, skip nếu timeout
- Output: `AgentResults` dataclass chứa tất cả agent outputs
- Logging: `logger.info(f"agents={activated}, latency={ms}ms")`

### 2.6 Update main loop

- Thêm router + agent_manager vào pipeline
- Template engine phải handle `AgentResults` (không chỉ `List[Detection]`)
- Extend template: nếu có action → thêm vào caption. Nếu có text → đọc text. Nếu có face → mô tả.

### Checkpoint Phase 2

Tự verify:
- Log cho thấy: static scene → chỉ object_agent chạy
- Đưa text vào camera → ocr_agent triggered
- Đưa mặt vào camera → face_agent triggered
- Di chuyển → action_agent triggered
- `pytest -q` pass, code quality checks pass

---

## Phase 3 — RAG Lite + Orchestrator

### 3.1 RAG module (`src/rag/knowledge_base.py`)

- **Check trước:** verify `models/minilm_l6.onnx` tồn tại
- Load MiniLM ONNX qua ONNX Runtime
- Tokenizer: `from transformers import AutoTokenizer` → `sentence-transformers/all-MiniLM-L6-v2`
- FAISS: `faiss.IndexFlatIP` (cosine similarity trên normalized vectors)
- Startup: load JSON files từ `data/knowledge/` → embed tất cả → build index
- `search(query: str, top_k: int = 3) -> List[Match]`
- Match dưới `min_similarity=0.3` → return empty

### 3.2 Knowledge base (`data/knowledge/`)

Tạo 3 JSON files, mỗi file ~25-30 entries, tiếng Việt có dấu, UTF-8:

**`hazard_patterns.json`** (~25 entries):
- car/truck/bus/motorcycle + vị trí + khoảng cách → cảnh báo tiếng Việt
- stairs + hướng → cảnh báo cầu thang
- Mỗi entry: `{"query": "...", "caption_vi": "...", "priority": 0|1}`

**`spatial_templates.json`** (~30 entries):
- object + direction + distance → mô tả tiếng Việt
- Mỗi entry: `{"query": "...", "caption_vi": "..."}`

**`relationship_patterns.json`** (~25 entries):
- object + object + action → mô tả relationship tiếng Việt
- Mỗi entry: `{"query": "...", "caption_vi": "..."}`

### 3.3 Orchestrator (`src/caption/orchestrator.py`)

- Thay thế template engine làm primary caption generator
- Input: `AgentResults` + `scene_types`
- Logic:
  1. Build query string từ agent outputs: `"{labels} {positions} {actions}"`
  2. RAG search → nếu match → dùng RAG caption
  3. Nếu no match → fallback về template engine (Phase 1, giữ nguyên, không xóa)
  4. Priority sort output: hazard first
  5. Dedup vs previous caption
  6. Log `source=rag|template` cho mỗi caption
- **KHÔNG XÓA template engine.** Giữ làm fallback. Orchestrator wrap template engine.

### 3.4 Config

```yaml
# Thêm vào configs/app.yaml
rag:
  enabled: true
  model_path: models/minilm_l6.onnx
  knowledge_dir: data/knowledge/
  top_k: 3
  min_similarity: 0.3
```

### Checkpoint Phase 3

Tự verify:
- `rag.enabled: true` → captions nguồn RAG
- `rag.enabled: false` → captions nguồn template, app vẫn chạy bình thường
- Query "person stairs" → return caption chứa "cầu thang"
- Query "nonsense xyz" → fallback template
- Log hiển thị `source=rag` hoặc `source=template`
- `pytest -q` pass

---

## Phase 4 — Threading + UX

### 4.1 Threading (`src/core/pipeline.py`)

- Refactor main loop thành threaded pipeline:
  - `CaptureThread`: push frames vào `queue.Queue(maxsize=5)`
  - `ProcessingThread`: consume frames → router → agents → orchestrator → push caption vào TTS queue
  - `TTSThread`: đã có từ Phase 1, consume caption queue
- `threading.Event()` cho shutdown signal
- Tất cả threads `daemon=True`
- `src/app.py` chỉ start threads + wait for KeyboardInterrupt
- **Không deadlock:** tất cả `queue.get(timeout=1.0)`, check shutdown event mỗi iteration

### 4.2 Hotkeys

- Dùng `keyboard` library hoặc `pynput`
- Chạy trong thread riêng
- Mappings từ config:
  ```yaml
  hotkeys:
    toggle: ctrl+shift+s
    describe_now: ctrl+shift+d
    read_text: ctrl+shift+t
    mute: ctrl+shift+m
  ```
- Audio feedback: mỗi hotkey → TTS đọc "Đã bật" / "Đã tắt" / "Đang mô tả"

### 4.3 Audio UX

- Startup: `tts.speak("Trợ lý đã sẵn sàng")`
- Shutdown: `tts.speak("Đã tắt")` → `time.sleep(2)` → exit
- Hazard: play short beep trước cảnh báo (dùng `numpy` generate sine wave 0.1s)
- Verbosity config: `low` (chỉ hazard) / `medium` (hazard + navigation) / `high` (tất cả)

### Checkpoint Phase 4

Tự verify:
- Chạy 5 phút liên tục, verify không crash
- `Ctrl+C` shutdown sạch, không treo threads
- Hotkeys hoạt động
- Hazard beep phát trước cảnh báo
- Memory stable (check `psutil.Process().memory_info().rss` đầu vs cuối)
- `pytest -q` pass

---

## Sau khi HOÀN THÀNH tất cả 4 phases

### Final code quality scan

```bash
black .
ruff check . --fix
ruff check . --select F401  # unused imports
mypy .
pytest -q
```

- Grep `src/` tìm: `TODO`, `FIXME`, `HACK`, `hardcode` → fix hoặc document
- Grep `"models/` và `"data/` → phải dùng config, không hardcode
- Verify mỗi file trong `src/` được import ít nhất 1 lần (không dead files)
- Verify mỗi function/class trong `src/` được gọi ít nhất 1 lần (không dead code)
- Verify không có circular imports: `python -c "import src.app"` không lỗi

### End-to-end verification

```bash
# Dry run — phải pass
python -m src.app --dry-run --config configs/app.yaml

# Live run — verify manually
python -m src.app --config configs/app.yaml
```

- Verify full pipeline: capture → router → agents → orchestrator/RAG → TTS → audio
- Verify change detection: camera tĩnh → không spam TTS
- Verify selective routing: chỉ agents cần thiết chạy
- Verify RAG vs template: log hiển thị cả 2 sources
- Verify hotkeys
- Verify graceful shutdown

### Docs update

- `README.md`: cập nhật Quick Start, requirements, cách chạy app
- `docs/setup.md`: install steps từ fresh clone → chạy được app
- `configs/app.yaml`: comment tiếng Việt giải thích mỗi field
- Verify tất cả paths trong docs match thực tế

### Final report

Khi hoàn thành, báo lại DUY NHẤT 1 LẦN với format:

```
## Kết quả
- Files created: [list]
- Files modified: [list]
- Tests: X passed, Y failed
- Lint: black ✓/✗, ruff ✓/✗, mypy ✓/✗
- Dry-run: pass/fail
- Live-run: tested/not tested
- Issues found & fixed: [list]
- Decisions made (không hỏi): [list with reasoning]
- Known limitations: [list]
```
