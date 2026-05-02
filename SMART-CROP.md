# Smart Crop — Thuật toán Cắt Ảnh Theo Chủ Đề (Cây)

## Tổng quan

Smart Crop là thuật toán cắt ảnh tự động thành hình vuông tỉ lệ 1:1, tập trung vào chủ thể chính là **cây/cây cối** trong ảnh. Thuật toán gồm 2 giai đoạn chính:

1. **Phát hiện cây** — sử dụng AI semantic segmentation
2. **Tính toán vùng cắt** — dựa trên bounding box của cây

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                   Smart Crop Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Ảnh gốc  │───▶│ Tree Detect  │───▶│  Crop Calc   │   │
│  │ (JPEG/PNG)│    │ (Python AI)  │    │  (TypeScript)│   │
│  └──────────┘    └──────────────┘    └──────────────┘   │
│                       │                      │           │
│                       ▼                      ▼           │
│              SegFormer Model         Sharp (extract)     │
│              Tree Bounding Box       1:1 Square Crop     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Thành phần

| File | Vai trò |
|------|---------|
| `detect_tree.py` | Python script — chạy AI model để phát hiện cây |
| `smart-crop.ts` | TypeScript — tính toán vùng cắt và lưu ảnh |
| `server.ts` | HTTP server — API crop đơn lẻ hoặc batch |

---

## Giai đoạn 1: Phát hiện cây (Tree Detection)

### Model AI

Sử dụng **SegFormer** — mô hình semantic segmentation của NVIDIA, được fine-tune trên dataset ADE20K.

- **Model**: `nvidia/segformer-b0-finetuned-ade-512-512`
- **Kích thước**: ~14MB (tự động tải lần đầu)
- **Framework**: PyTorch + HuggingFace Transformers

### Classes được nhận diện

| Class ID | Tên | Ghi chú |
|----------|-----|---------|
| `4` | tree | Lớp cây chính |
| `17` | plant, flora | Cây bụi, thực vật |
| `9` | grass | **Không dùng** — chỉ tree + plant |

```python
TREE_CLASSES = {4, 17}  # Chỉ 2 class này được coi là "cây"
```

### Quy trình phát hiện

```
Ảnh đầu vào
    │
    ▼
┌─────────────────────────────────┐
│ 1. SegFormer Segmentation       │
│    - Resize về 512x512          │
│    - Chạy model inference       │
│    - Upsample về kích thước gốc │
│    - Output: class map (H x W)  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 2. Tạo binary mask              │
│    - Lọc pixel thuộc TREE_CLASSES│
│    - Loại bỏ nếu diện tích < 0.5%│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 3. Connected Components         │
│    - scipy.ndimage.label()      │
│    - Tìm các vùng liên tiếp     │
│    - Tính bounding box mỗi vùng │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 4. Thêm padding 5%              │
│    - Mở rộng box 5% mỗi chiều   │
│    - Clamp trong biên ảnh       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 5. Sort theo diện tích          │
│    - Vùng lớn nhất = primary    │
│    - Trả về tất cả box tìm được │
└─────────────────────────────────┘
```

### Output format

```json
{
  "file": "/path/to/image.jpg",
  "width": 1920,
  "height": 1080,
  "tree_found": true,
  "tree_box": { "x": 400, "y": 100, "w": 800, "h": 900 },
  "tree_coverage": 0.35,
  "all_tree_boxes": [
    { "x": 400, "y": 100, "w": 800, "h": 900 },
    { "x": 1200, "y": 200, "w": 300, "h": 400 }
  ]
}
```

### Tham số quan trọng

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `BOX_PADDING` | `0.05` | Padding 5% quanh bounding box |
| `MIN_TREE_AREA_FRACTION` | `0.005` | Diện tích tối thiểu = 0.5% ảnh |

---

## Giai đoạn 2: Tính toán vùng cắt (Crop Calculation)

### Thuật toán

```typescript
function computeSquareCrop(imageWidth, imageHeight, treeBox): CropRect
```

#### Trường hợp 1: Có phát hiện cây

```
Bước 1: Tính kích thước ô vuông mục tiêu
  targetSize = max(treeBox.w, treeBox.h) × BREATHING_ROOM
  BREATHING_ROOM = 1.2 (padding 20%)

Bước 2: Giới hạn trong kích thước ảnh
  targetSize = min(targetSize, min(imageWidth, imageHeight))

Bước 3: Tâm crop = tâm của tree box
  cropX = treeCenterX - targetSize / 2
  cropY = treeCenterY - targetSize / 2

Bước 4: Clamp trong biên ảnh
  cropX = max(0, min(cropX, imageWidth - targetSize))
  cropY = max(0, min(cropY, imageHeight - targetSize))
```

#### Trường hợp 2: Không phát hiện cây (Fallback)

```
  targetSize = min(imageWidth, imageHeight)
  cropX = (imageWidth - targetSize) / 2   // center crop
  cropY = (imageHeight - targetSize) / 2
```

### Ví dụ minh họa

```
Ảnh gốc: 1920 x 1080
Tree box: x=400, y=100, w=800, h=900

1. targetSize = max(800, 900) × 1.2 = 1080
2. maxSize = min(1920, 1080) = 1080 → targetSize = 1080
3. treeCenterX = 400 + 800/2 = 800
   treeCenterY = 100 + 900/2 = 550
   cropX = 800 - 1080/2 = 260
   cropY = 550 - 1080/2 = 10
4. Clamp: cropX=260, cropY=10 (hợp lệ)

Kết quả: Cắt ô vuông 1080x1080 tại (260, 10)
```

### Visualization

```
┌──────────────────────────────────────────────────────┐
│                      Ảnh gốc                          │
│  ┌──────────────────────────────────────────────┐    │
│  │                                              │    │
│  │         ┌─────────────────────┐              │    │
│  │         │                     │              │    │
│  │         │    🌳 Cây (tree)    │              │    │
│  │         │                     │              │    │
│  │         └─────────────────────┘              │    │
│  │              tree_box                         │    │
│  │                                              │    │
│  │    ┌─────────────────────────────┐           │    │
│  │    │                             │           │    │
│  │    │   ┌───────────────────┐     │           │    │
│  │    │   │                   │     │           │    │
│  │    │   │   Vùng crop 1:1   │     │           │    │
│  │    │   │   (output)        │     │           │    │
│  │    │   │                   │     │           │    │
│  │    │   └───────────────────┘     │           │    │
│  │    │   breathing_room = 1.2x     │           │    │
│  │    │                             │           │    │
│  │    └─────────────────────────────┘           │    │
│  │                                              │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## Giai đoạn 3: Cắt và lưu ảnh

Sử dụng thư viện **sharp** để extract vùng crop:

```typescript
await sharp(sourceFile)
  .extract({
    left: crop.x,
    top: crop.y,
    width: crop.size,
    height: crop.size,
  })
  .jpeg({ quality: 92 })
  .toFile(outputFile);
```

- **Định dạng output**: JPEG
- **Chất lượng**: 92%
- **Tên file**: `{id}_crop.jpg`

---

## Cách sử dụng

### CLI (smart-crop.ts)

```bash
# Xử lý tất cả ảnh chưa crop
bun run smart-crop.ts

# Xử lý ảnh cụ thể theo ID
bun run smart-crop.ts --ids 1 2 3

# Xử lý N ảnh đầu tiên
bun run smart-crop.ts --limit 10

# Crop lại ảnh đã crop
bun run smart-crop.ts --reprocess
```

### API (server.ts)

```bash
# Crop 1 ảnh
POST /api/smart-crop/:id

# Crop batch (background)
POST /api/smart-crop-all
```

---

## Lưu trữ Database

Bảng `images` lưu thông tin crop:

| Column | Type | Mô tả |
|--------|------|-------|
| `crop_filename` | TEXT | Tên file ảnh đã crop |
| `crop_x` | INTEGER | Tọa độ X của vùng crop |
| `crop_y` | INTEGER | Tọa độ Y của vùng crop |
| `crop_size` | INTEGER | Kích thước ô vuông crop |
| `tree_box_json` | TEXT | JSON bounding box của cây |

---

## Batch Processing

Để tối ưu hiệu năng, ảnh được xử lý theo batch:

- **Batch size**: 20 ảnh/lần gọi Python
- **Reason**: Tránh load model nhiều lần, model chỉ load 1 lần cho cả batch

```
Batch 1: [img1, img2, ..., img20] → 1× python3 detect_tree.py
Batch 2: [img21, img22, ..., img40] → 1× python3 detect_tree.py
...
```

---

## Flow tổng thể

```
1. Lấy danh sách ảnh cần crop từ DB
2. Chia thành batch (20 ảnh/batch)
3. Với mỗi batch:
   a. Gọi detect_tree.py → nhận tree boxes
   b. Với mỗi ảnh:
      - computeSquareCrop() → tính crop rect
      - sharp.extract() → cắt ảnh
      - Lưu vào images-cropped/
      - Update DB với thông tin crop
4. In thống kê: thành công, thất bại, không có cây
```

---

## Tham số cấu hình

| Hằng số | Giá trị | File | Mô tả |
|---------|---------|------|-------|
| `BREATHING_ROOM` | `1.2` | smart-crop.ts | Padding 20% quanh cây |
| `DETECT_BATCH_SIZE` | `20` | smart-crop.ts | Số ảnh mỗi batch |
| `BOX_PADDING` | `0.05` | detect_tree.py | Padding 5% trong detection |
| `MIN_TREE_AREA_FRACTION` | `0.005` | detect_tree.py | Diện tích cây tối thiểu |
| `JPEG_QUALITY` | `92` | smart-crop.ts | Chất lượng ảnh output |
