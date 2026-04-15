# Demo UI — Implementation Prompt

## SYSTEM RULES

1. **ĐỌC TRƯỚC.** Scan toàn bộ `src/` hiểu pipeline hiện tại trước khi code. Đặc biệt đọc `src/app.py`, `src/core/pipeline.py`, `src/core/types.py`, `src/agents/`, `src/caption/`.
2. **KHÔNG DUPLICATE.** Không tạo lại logic detect/caption/tts. UI chỉ consume data từ pipeline hiện có.
3. **KHÔNG PHÁ PIPELINE.** App phải chạy được cả 2 mode: `--headless` (audio only, cho người khiếm thị) và `--visual` (có UI, cho demo/debug). Default là `--headless`.
4. **SELF-TEST.** Chạy `black .`, `ruff check . --fix`, `mypy .`, `pytest -q` trước khi báo.
5. **KHÔNG HỎI.** Đủ chi tiết rồi, tự quyết định, ghi lại trong commit.

---

## Mục tiêu

Thêm visual demo UI để:
- Demo trước thầy/lớp: thầy nhìn thấy camera feed + bounding boxes + caption
- Debug: dev thấy agents nào đang chạy, latency, RAG vs template
- KHÔNG thay đổi core pipeline — UI chỉ là viewer, đọc data từ pipeline

---

## Cấu trúc file

```
src/
├── ui/
│   ├── __init__.py
│   ├── demo_window.py       # Main demo UI window
│   ├── overlay_renderer.py  # Vẽ bbox, labels, caption lên frame
│   └── dashboard.py         # Panel hiển thị metrics/status
```

KHÔNG dùng file name có phase/week/version prefix.

---

## 1. Overlay renderer (`src/ui/overlay_renderer.py`)

Vẽ trực tiếp lên frame numpy array bằng OpenCV. Không cần library UI nào khác.

### Bounding boxes

```python
def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
```

- Mỗi detection: vẽ `cv2.rectangle()` + label + confidence
- Màu theo priority:
  - P0 hazard (car, truck, bus, motorcycle): **đỏ** `(0, 0, 255)`
  - P1 navigation (stairs, traffic_light, stop_sign): **vàng** `(0, 255, 255)`
  - P2 info (person, dog, bench, chair): **xanh lá** `(0, 255, 0)`
- Label format: `"person 0.87"` — tên class + confidence
- Font: `cv2.FONT_HERSHEY_SIMPLEX`, scale 0.6, thickness 2
- Label background: filled rectangle phía trên bbox để text đọc được

### Spatial indicators

- Vẽ text position + distance cạnh bbox: "trái, gần"
- Font nhỏ hơn, scale 0.5, màu trắng

### Action overlay

- Nếu có action result: vẽ text góc trên phải frame
- Format: `"Action: walking (0.72)"`
- Background semi-transparent

### OCR overlay

- Nếu có text regions: vẽ bbox quanh text, hiển thị nội dung text đọc được
- Màu: **cyan** `(255, 255, 0)`

### Face overlay

- Nếu có face results: vẽ bbox quanh face, hiển thị emotion
- Màu: **magenta** `(255, 0, 255)`

---

## 2. Dashboard panel (`src/ui/dashboard.py`)

Vẽ status bar ở dưới frame hoặc bên phải. Vẫn dùng OpenCV, không cần Qt/Tkinter.

```python
def draw_dashboard(frame: np.ndarray, status: PipelineStatus) -> np.ndarray:
```

### Thông tin hiển thị

- **Caption hiện tại:** 2 dòng text lớn ở bottom frame
  - Dòng 1: caption tiếng Việt
  - Dòng 2 nhỏ hơn: `[RAG]` hoặc `[Template]` + source indicator

- **Status bar** (thanh ngang dưới cùng, background đen semi-transparent):
  - FPS: `"2.1 FPS"`
  - Latency: `"Pipeline: 45ms"`
  - Active agents: `"Agents: OBJ OCR"` (viết tắt agents đang chạy)
  - Scene type: `"Scene: text_present"`
  - Detection count: `"Det: 3"`

- **Verbosity indicator** góc trên trái:
  - `"[LOW]"` / `"[MED]"` / `"[HIGH]"`

- **Mute indicator** nếu đang mute:
  - `"🔇 MUTED"` góc trên phải (dùng text, không dùng emoji trong code — ghi "MUTED")

### Layout

```
┌──────────────────────────────────────────┐
│ [HIGH]                          [MUTED]  │
│                                          │
│     ┌──────────┐                         │
│     │ person   │    Action: walking      │
│     │ 0.92     │                         │
│     └──────────┘                         │
│                                          │
│  ┌────────────────────────────────────┐   │
│  │ Phía trước có 1 người đang đi bộ  │   │
│  │ [RAG] 3 agents | 42ms             │   │
│  └────────────────────────────────────┘   │
│  FPS: 2.1 | Pipeline: 45ms | Det: 3     │
└──────────────────────────────────────────┘
```

---

## 3. Demo window (`src/ui/demo_window.py`)

### Main class

```python
class DemoWindow:
    def __init__(self, config: dict):
        self.window_name = "Accessibility Vision Assistant"
        self.running = True
    
    def show_frame(self, frame: np.ndarray, 
                   detections: List[Detection],
                   agent_results: AgentResults,
                   caption: str,
                   status: PipelineStatus):
        """Vẽ overlay + dashboard lên frame rồi hiển thị."""
        display = frame.copy()
        display = draw_detections(display, detections)
        display = draw_dashboard(display, status)
        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
    
    def close(self):
        cv2.destroyAllWindows()
```

### Keyboard controls trong UI window

- `q`: quit app
- `d`: force describe (giống hotkey Ctrl+Shift+D)
- `m`: mute/unmute
- `v`: cycle verbosity (low → med → high → low)
- `r`: toggle RAG on/off
- `s`: screenshot — save current frame + overlay to `screenshots/`

### Window resize

- Default: 960x720 hoặc camera native resolution
- Cho phép resize window, content scale theo
- `cv2.namedWindow(name, cv2.WINDOW_NORMAL)`

---

## 4. Tích hợp vào pipeline (`src/app.py` và `src/core/pipeline.py`)

### Config

```yaml
# Thêm vào configs/app.yaml
ui:
  enabled: false              # default headless (audio only)
  window_width: 960
  window_height: 720
  show_fps: true
  show_detections: true
  show_caption: true
  show_dashboard: true
  screenshot_dir: screenshots/
```

### CLI flags

```bash
# Headless (default, cho người khiếm thị)
python -m src.app

# Visual mode (cho demo/debug)
python -m src.app --visual

# Visual + headless cùng lúc (UI + TTS)
python -m src.app --visual --tts

# Dry run với visual
python -m src.app --visual --dry-run
```

- `--visual` override `ui.enabled` thành `true`
- Không có `--visual` → app chạy headless như trước, KHÔNG BỊ ẢNH HƯỞNG

### Pipeline integration

```python
# Trong ProcessingThread hoặc main loop:
# SAU KHI đã có detections, caption, status:

if ui_enabled:
    demo_window.show_frame(frame, detections, agent_results, caption, status)

# TTS vẫn chạy bình thường song song, KHÔNG phụ thuộc UI
```

### PipelineStatus dataclass

Thêm vào `src/core/types.py`:

```python
@dataclass
class PipelineStatus:
    fps: float
    pipeline_latency_ms: float
    active_agents: List[str]
    scene_type: str
    detection_count: int
    caption_source: str  # "rag" | "template"
    verbosity: str       # "low" | "medium" | "high"
    is_muted: bool
```

- Pipeline cập nhật `PipelineStatus` mỗi frame
- UI và TTS đều consume từ cùng data, không duplicate logic

---

## 5. Screenshot feature

```python
def save_screenshot(frame: np.ndarray, caption: str, output_dir: str = "screenshots/"):
    """Lưu frame + overlay khi user nhấn 's' hoặc khi muốn capture demo."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(filepath, frame)
    # Lưu caption vào text file cùng tên
    with open(filepath.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        f.write(caption)
```

- Dùng cho báo cáo LaTeX: screenshot demo + caption → paste vào paper
- `screenshots/` thêm vào `.gitignore`

---

## 6. Tests

### `tests/test_ui.py`

```python
def test_overlay_renderer_draws_bbox():
    """Tạo blank frame, add detection, verify frame modified."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(label="person", confidence=0.9, bbox=(100,100,200,200), position="trước", distance="gần")
    result = draw_detections(frame, [det])
    assert result.sum() > 0  # frame không còn blank

def test_dashboard_draws_status():
    """Verify dashboard adds status bar."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    status = PipelineStatus(fps=2.0, pipeline_latency_ms=45, ...)
    result = draw_dashboard(frame, status)
    assert result.sum() > 0

def test_demo_window_headless_mode():
    """Verify app chạy bình thường khi ui.enabled=false."""
    # Import app, chạy dry-run, verify không crash, không import cv2.imshow
    pass

def test_screenshot_creates_files(tmp_path):
    """Verify screenshot saves png + txt."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    save_screenshot(frame, "test caption", str(tmp_path))
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert len(list(tmp_path.glob("*.txt"))) == 1
```

---

## Verification checklist

- [ ] `python -m src.app --visual --dry-run` → window hiện, bbox vẽ, caption hiển thị, tự tắt sau 10 frames
- [ ] `python -m src.app --visual` → live webcam feed với overlays
- [ ] `python -m src.app` (không có --visual) → chạy headless như trước, KHÔNG bị ảnh hưởng gì
- [ ] Nhấn `s` trong UI → screenshot saved
- [ ] Nhấn `m` → mute indicator hiện
- [ ] Nhấn `v` → verbosity cycle
- [ ] Nhấn `q` → quit sạch
- [ ] Bbox màu đúng: đỏ=hazard, vàng=navigation, xanh=info
- [ ] Caption hiển thị tiếng Việt có dấu đúng (font OpenCV hỗ trợ ASCII only → dùng `cv2.putText` với `cv2.FONT_HERSHEY_SIMPLEX`, tiếng Việt sẽ bị lỗi font → xử lý bằng `PIL.ImageFont` + `PIL.ImageDraw` rồi convert back numpy)
- [ ] Dashboard hiện FPS, latency, active agents realtime
- [ ] `pytest -q` pass
- [ ] `black .`, `ruff check .`, `mypy .` pass
- [ ] Không có dead code, unused imports
- [ ] Headless mode không import `cv2.imshow` — lazy import only khi `--visual`

---

## Lưu ý tiếng Việt trong OpenCV

OpenCV `putText` KHÔNG hỗ trợ Unicode/tiếng Việt. Cần workaround:

```python
from PIL import Image, ImageDraw, ImageFont

def put_vietnamese_text(frame: np.ndarray, text: str, position: tuple, 
                        font_size: int = 20, color: tuple = (255,255,255)) -> np.ndarray:
    """Vẽ text tiếng Việt lên frame bằng PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Windows
    except OSError:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

Dùng function này thay `cv2.putText` cho tất cả text tiếng Việt (caption, labels, status).

---

## Final report format

```
## Kết quả Demo UI
- Files created: [list]
- Files modified: [list]
- Tests: X passed, Y failed
- Lint: black ✓/✗, ruff ✓/✗, mypy ✓/✗
- Visual mode: tested/not tested
- Headless mode: vẫn hoạt động ✓/✗
- Screenshot feature: ✓/✗
- Vietnamese text rendering: ✓/✗
- Known limitations: [list]
```
