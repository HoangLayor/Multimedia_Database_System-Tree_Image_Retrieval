## 2.2. Tiền xử lý ảnh

Tiền xử lý ảnh là một bước can thiệp kỹ thuật mang tính nền tảng trong mọi hệ thống thị giác máy tính, đóng vai trò bản lề nhằm chuẩn hóa không gian dữ liệu đầu vào, khoanh vùng chính xác khu vực chứa đối tượng mục tiêu và tạo tiền đề cho các thao tác cắt ảnh tự động downstream. Đối với đặc thù của bài toán nhận diện và cô lập cây trong ảnh phong cảnh, hình ảnh thu thập thường chịu ảnh hưởng mạnh bởi điều kiện môi trường chiếu sáng, bóng đổ, các cấu trúc hậu cảnh phức tạp và sự hiện diện đồng thời của nhiều vật thể cạnh tranh. Nhằm giải quyết triệt để những thách thức này, hệ thống đã thiết kế và triển khai một luồng xử lý hoàn toàn tự động, tích hợp ba kỹ thuật cốt lõi được tối ưu hóa khắt khe cả về mặt cơ sở toán học lẫn logic lập trình: phân đoạn ngữ nghĩa bằng mô hình SegFormer, phân tích thành phần liên thông để xác định bounding box, và tính toán vùng cắt hình vuông 1:1 tối ưu.

### 2.2.1. Phân đoạn ngữ nghĩa với mô hình SegFormer

Hệ thống ưu tiên sử dụng mô hình SegFormer — kiến trúc semantic segmentation hiện đại do NVIDIA phát triển, được fine-tune trên bộ dataset ADE20K gồm 150 lớp đối tượng — để tiến hành phân loại từng điểm ảnh trong không gian ảnh đầu vào. Căn nguyên của sự lựa chọn này nằm ở việc SegFormer kết hợp thành công giữa Transformer encoder và lightweight MLP decoder, cho khả năng nắm bắt ngữ cảnh toàn cục (global context) vượt trội so với các kiến trúc CNN truyền thống, đồng thời duy trì hiệu suất inference ở mức chấp nhận được ngay cả trên phần cứng CPU.

Về mặt toán học, quá trình phân đoạn được thực hiện qua chuỗi biến đổi như sau. Gọi ảnh đầu vào là ma trận cường độ điểm ảnh 𝐼 ∈ ℝ^(𝐻×𝑊×3) với 𝐻, 𝑊 lần lượt là chiều cao và chiều rộng. Bộ xử lý ảnh SegformerImageProcessor thực hiện phép ánh xạ chuẩn hóa:

𝐼_𝑛𝑜𝑟𝑚 = Normalize(Resize(𝐼, 512×512))

Trong đó, phép Resize nội suy ảnh về kích thước cố định 512×512 — kích thước đầu vào tiêu chuẩn của mô hình — thông qua phép nội suy bilinear, và Normalize thực hiện phép chuẩn hóa từng kênh màu dựa trên giá trị trung bình μ và độ lệch chuẩn σ được học từ tập huấn luyện:

𝐼_𝑐ℎ𝑎𝑛𝑛𝑒𝑙 = (𝐼_𝑐ℎ𝑎𝑛𝑛𝑒𝑙 − μ) / σ

Mô hình SegFormer sau đó thực thi phép biến đổi đặc trưng thông qua Transformer encoder gồm 𝐿 lớp attention, sinh ra chuỗi bản đồ đặc trưng đa tỷ lệ {𝐹₁, 𝐹₂, ..., 𝐹_𝐿} với độ phân giải giảm dần. MLP decoder hợp nhất các bản đồ này thông qua phép nội suy lên cùng độ phân giải 𝐻/4 × 𝑊/4, sau đó áp dụng phép chập 1×1 để ánh xạ vào không gian số lớp 𝐶:

logits = Conv₁×₁(MLP_Decoder([𝐹₁, 𝐹₂, ..., 𝐹_𝐿]))

Kết quả đầu ra là tensor logits ∈ ℝ^(1×𝐶×𝐻/4×𝑊/4) chứa điểm số chưa chuẩn hóa cho mỗi lớp tại mỗi vị trí không gian. Bản đồ phân đoạn cuối cùng được xác định bằng phép argmax dọc theo chiều lớp:

𝑆(𝑖, 𝑗) = argmax_c(logits[𝑐, 𝑖, 𝑗])

Để khôi phục bản đồ phân đoạn về độ phân giải gốc của ảnh đầu vào, hệ thống áp dụng phép nội suy bilinear lên toàn bộ tensor logits trước khi thực thi argmax:

𝑆_𝑓𝑢𝑙𝑙 = argmax_c(Interpolate_bilinear(logits, (𝐻, 𝑊))[𝑐, 𝑖, 𝑗])

Trong bài toán nhận diện cây, hệ thống chỉ quan tâm đến hai lớp đối tượng trong tập ADE20K: lớp "tree" (class ID = 4) đại diện cho thân cây và tán cây lớn, cùng lớp "plant, flora" (class ID = 17) bao gồm cây bụi và thực vật thân thấp. Mặt nạ nhị phân 𝑀 được xây dựng bằng phép hợp của hai lớp này:

𝑀(𝑖, 𝑗) = 1 nếu 𝑆_𝑓𝑢𝑙𝑙(𝑖, 𝑗) ∈ {4, 17}, ngược lại 𝑀(𝑖, 𝑗) = 0

Lớp "grass" (class ID = 9) được chủ đích loại khỏi tập xét nhằm tránh nhiễu từ mặt cỏ nền, đảm bảo mặt nạ chỉ phản ánh cấu trúc cây thẳng đứng — đối tượng mục tiêu của bài toán crop ảnh.

### 2.2.2. Phát hiện vùng cây qua phân tích thành phần liên thông

Sau khi thu được mặt nạ nhị phân 𝑀 từ bước phân đoạn ngữ nghĩa, hệ thống tiến hành phân tích các vùng liên thông (connected components) nhằm tách biệt từng cụm cây độc lập và xác định bounding box bao quanh mỗi cụm. Phương pháp này cho phép hệ thống không chỉ phát hiện sự hiện diện của cây mà còn định vị chính xác tọa độ không gian của từng thực thể trong khung hình.

Về mặt lý thuyết, phép gán nhãn thành phần liên thông được thực thi thông qua thuật toán flood-fill hai chiều trên mặt nạ 𝑀. Gọi 𝐿 là ma trận nhãn có cùng kích thước với 𝑀, khởi tạo toàn bộ bằng 0. Với mỗi điểm ảnh (𝑖, 𝑗) thỏa 𝑀(𝑖, 𝑗) = 1 và chưa được gán nhãn, hệ thống thực thi phép lan truyền vùng: gán nhãn hiện tại cho (𝑖, 𝑗) và đệ quy gán cho tất cả các láng giềng 4-hoặc-8-liên thông cũng thỏa điều kiện 𝑀 = 1. Quá trình lặp cho đến khi duyệt exhaustively toàn bộ không gian ảnh:

𝐿(𝑖, 𝑗) = 𝑘 nếu (𝑖, 𝑗) thuộc thành phần liên thông thứ 𝑘

Số lượng thành phần liên thông 𝐾 thu được phản ánh số lượng cụm cây độc lập trong ảnh. Tuy nhiên, không phải mọi thành phần đều đại diện cho cây thực sự — các vùng nhiễu nhỏ hoặc phân đoạn sai có thể tạo thành các component có diện tích không đáng kể. Nhằm loại bỏ các trường hợp nhiễu này, hệ thống áp dụng ngưỡng diện tích tối thiểu dựa trên tỷ lệ diện tích so với tổng diện tích ảnh:

𝐴_𝑚𝑖𝑛 = 𝛼 × 𝐻 × 𝑊

Với 𝛼 = 0.005 (tương đương 0.5% diện tích ảnh) — ngưỡng được tinh chỉnh thực nghiệm để cân bằng giữa độ nhạy phát hiện cây nhỏ và khả năng loại nhiễu. Chỉ những thành phần liên thông có diện tích 𝐴_𝑘 ≥ 𝐴_𝑚𝑖𝑛 mới được giữ lại cho các bước xử lý downstream.

Với mỗi thành phần liên thông hợp lệ thứ 𝑘, bounding box được xác định bằng phép chiếu lên hai trục tọa độ:

𝑥_𝑚𝑖𝑛 = min{𝑗 | 𝐿(𝑖, 𝑗) = 𝑘}
𝑥_𝑚𝑎𝑥 = max{𝑗 | 𝐿(𝑖, 𝑗) = 𝑘}
𝑦_𝑚𝑖𝑛 = min{𝑖 | 𝐿(𝑖, 𝑗) = 𝑘}
𝑦_𝑚𝑎𝑥 = max{𝑖 | 𝐿(𝑖, 𝑗) = 𝑘}

Từ đó suy ra chiều rộng 𝑤 = 𝑥_𝑚𝑎𝑥 − 𝑥_𝑚𝑖𝑛 + 1 và chiều cao ℎ = 𝑦_𝑚𝑎𝑥 − 𝑦_𝑚𝑖𝑛 + 1 của bounding box. Nhằm đảm bảo vùng crop không cắt sát vào biên cây — tránh hiện tượng mất mát thông tin biên do sai số phân đoạn — hệ thống áp dụng thêm lớp padding mở rộng bounding box theo tỷ lệ 𝛽:

𝑝𝑎𝑑_𝑥 = ⌊𝛽 × 𝑤⌋
𝑝𝑎𝑑_𝑦 = ⌊𝛽 × ℎ⌋

𝑥_𝑚𝑖𝑛 = max(0, 𝑥_𝑚𝑖𝑛 − 𝑝𝑎𝑑_𝑥)
𝑦_𝑚𝑖𝑙 = max(0, 𝑦_𝑚𝑖𝑛 − 𝑝𝑎𝑑_𝑦)
𝑥_𝑚𝑎𝑥 = min(𝑊 − 1, 𝑥_𝑚𝑎𝑥 + 𝑝𝑎𝑑_𝑥)
𝑦_𝑚𝑎𝑥 = min(𝐻 − 1, 𝑦_𝑚𝑎𝑥 + 𝑝𝑎𝑑_𝑦)

Với 𝛽 = 0.05 (padding 5%) — giá trị đủ để tạo vùng đệm an toàn quanh biên cây mà không làm loãng quá nhiều vùng crop. Sau bước padding, bounding box được cập nhật với 𝑤_𝑛𝑒𝑤 = 𝑥_𝑚𝑎𝑥 − 𝑥_𝑚𝑖𝑛 + 1 và ℎ_𝑛𝑒𝑤 = 𝑦_𝑚𝑎𝑥 − 𝑦_𝑚𝑖𝑛 + 1.

Cuối cùng, danh sách bounding box được sắp xếp giảm dần theo diện tích thành phần liên thông tương ứng, sao cho bounding box đầu tiên — tương ứng với cụm cây lớn nhất — được xem là primary tree box phục vụ cho bước tính toán vùng cắt. Tỷ lệ phủ cây (tree coverage) được tính bằng tổng diện tích tất cả thành phần liên thông chia cho tổng diện tích ảnh:

coverage = (Σ_{𝑘=1}^{𝐾} 𝐴_𝑘) / (𝐻 × 𝑊)

Chỉ số này cung cấp thước đo định lượng về mức độ hiện diện của cây trong khung hình, hữu ích cho việc lọc và phân loại ảnh downstream.

### 2.2.3. Tính toán vùng cắt và crop ảnh thông minh

Mục tiêu tối thượng của giai đoạn tiền xử lý là sinh ra một vùng cắt hình vuông tỷ lệ 1:1, tập trung vào chủ thể cây với khoảng thở (breathing room) hợp lý xung quanh. Yêu cầu này xuất phát từ đặc thù của các mô hình sinh ảnh downstream — vốn thường đòi hỏi đầu vào có kích thước vuông và bố cục cân đối.

Trường hợp thứ nhất: khi hệ thống phát hiện thành công ít nhất một vùng cây (tree_found = true). Gọi bounding box chính là 𝐵 = (𝑡𝑥, 𝑡𝑦, 𝑡𝑤, 𝑡ℎ) với (𝑡𝑥, 𝑡𝑦) là tọa độ góc trên-trái và (𝑡𝑤, 𝑡ℎ) là chiều rộng và chiều cao. Kích thước ô vuông mục tiêu được xác định dựa trên chiều lớn nhất của bounding box, nhân với hệ số khoảng thở 𝛾:

𝑆_𝑡𝑎𝑟𝑔𝑒𝑡 = ⌈max(𝑡𝑤, 𝑡ℎ) × 𝛾⌉

Với 𝛾 = 1.2 (khoảng thở 20%) — giá trị được tinh chỉnh để đảm bảo cây không bị cắt sát biên, đồng thời giữ tỷ lệ cây trong khung ở mức thẩm mỹ. Nhằm đảm bảo ô vuông không vượt quá biên ảnh, kích thước mục tiêu được giới hạn trên bởi chiều ngắn nhất của ảnh:

𝑆_𝑓𝑖𝑛𝑎𝑙 = min(𝑆_𝑡𝑎𝑟𝑔𝑒𝑡, min(𝑊, 𝐻))

Tâm của vùng cắt được căn chỉnh với tâm của bounding box cây:

𝑐𝑥 = 𝑡𝑥 + 𝑡𝑤 / 2
𝑐𝑦 = 𝑡𝑦 + 𝑡ℎ / 2

Từ đó suy ra tọa độ góc trên-trái của vùng cắt:

𝑥_𝑐𝑟𝑜𝑝 = ⌊𝑐𝑥 − 𝑆_𝑓𝑖𝑛𝑎𝑙 / 2⌋
𝑦_𝑐𝑟𝑜𝑝 = ⌊𝑐𝑦 − 𝑆_𝑓𝑖𝑛𝑎𝑙 / 2⌋

Để xử lý trường hợp vùng cắt vượt ra ngoài biên ảnh, hệ thống áp dụng phép clamp:

𝑥_𝑐𝑟𝑜𝑝 = max(0, min(𝑥_𝑐𝑟𝑜𝑝, 𝑊 − 𝑆_𝑓𝑖𝑛𝑎𝑙))
𝑦_𝑐𝑟𝑜𝑝 = max(0, min(𝑦_𝑐𝑟𝑜𝑝, 𝐻 − 𝑆_𝑓𝑖𝑛𝑎𝑙))

Phép clamp đảm bảo vùng cắt luôn nằm hoàn toàn trong biên ảnh, tránh hiện tượng lỗi truy cập bộ nhớ hoặc padding đen khi crop.

Trường hợp thứ hai: khi hệ thống không phát hiện được vùng cây nào thỏa ngưỡng (tree_found = false). Trong tình huống này, hệ thống kích hoạt cơ chế fallback — thực hiện phép cắt trung tâm (center crop) trên toàn bộ ảnh:

𝑆_𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘 = min(𝑊, 𝐻)
𝑥_𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘 = ⌊(𝑊 − 𝑆_𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘) / 2⌋
𝑦_𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘 = ⌊(𝐻 − 𝑆_𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘) / 2⌋

Cơ chế fallback đảm bảo hệ thống luôn sinh ra được ảnh output hợp lệ ngay cả trong trường hợp detection thất bại, duy trì tính liên tục của pipeline xử lý hàng loạt.

Sau khi xác định được vùng cắt (𝑥_𝑐𝑟𝑜𝑝, 𝑦_𝑐𝑟𝑜𝑝, 𝑆_𝑓𝑖𝑛𝑎𝑙), hệ thống sử dụng thư viện sharp — engine xử lý ảnh hiệu năng cao xây dựng trên libvips — để thực thi thao tác extract:

𝐼_𝑐𝑟𝑜𝑝𝑝𝑒𝑑 = sharp.extract(𝐼, {left: 𝑥_𝑐𝑟𝑜𝑝, top: 𝑦_𝑐𝑟𝑜𝑝, width: 𝑆_𝑓𝑖𝑛𝑎𝑙, height: 𝑆_𝑓𝑖𝑛𝑎𝑙})

Ảnh kết quả được mã hóa sang định dạng JPEG với hệ số chất lượng 𝑞 = 92% — ngưỡng cân bằng giữa kích thước file và độ trung thực thị giác — trước khi lưu vào thư mục output. Tên file được đặt theo quy chuẩn {𝑖𝑑}_𝑐𝑟𝑜𝑝.𝑗𝑝𝑔 với 𝑖𝑑 là định danh duy nhất của ảnh trong database.

Toàn bộ thông tin crop — bao gồm tên file output, tọa độ (𝑥, 𝑦), kích thước 𝑆, và JSON bounding box gốc — được cập nhật ngược vào bảng images trong SQLite, tạo điều kiện cho việc truy xuất, thống kê và tái xử lý sau này.

### 2.2.4. Tối ưu hóa batch processing

Để xử lý hiệu quả khối lượng ảnh lớn trong thực tế, hệ thống triển khai chiến lược batch processing nhằm tối đa hóa hiệu suất inference của mô hình SegFormer. Thay vì gọi riêng lẻ script detect_tree.py cho từng ảnh — dẫn đến overhead load model lặp lại (~14MB RAM cho mỗi invocation) — hệ thống gom nhóm ảnh thành từng batch có kích thước cố định 𝑁 = 20 và truyền toàn bộ đường dẫn ảnh làm tham số dòng lệnh cho một lần gọi Python duy nhất:

python3 detect_tree.py img₁.jpg img₂.jpg ... img_𝑁.jpg

Trong nội bộ detect_tree.py, mô hình SegFormer được load một lần duy nhất vào bộ nhớ, sau đó inference tuần tự trên từng ảnh trong batch. Chiến lược này giảm đáng kể tổng thời gian xử lý: nếu gọi đơn lẻ tốn 𝑇_𝑙𝑜𝑎𝑑 + 𝑇_𝑖𝑛𝑓𝑒𝑟𝑒𝑛𝑐𝑒 cho mỗi ảnh, thì batch processing chỉ tốn 𝑇_𝑙𝑜𝑎𝑑 + 𝑁 × 𝑇_𝑖𝑛𝑓𝑒𝑟𝑒𝑛𝑐𝑒 cho 𝑁 ảnh, tương đương tiết kiệm (𝑁−1) × 𝑇_𝑙𝑜𝑎𝑑.

Kết quả detection của toàn bộ batch được tuần tự hóa thành JSON array và xuất ra stdout, trong khi progress log được ghi vào stderr để tránh xung đột stream. TypeScript process đọc stdout, parse JSON và ánh xạ kết quả về từng ảnh dựa trên đường dẫn file, sau đó tuần tự thực thi crop và update database.

Chiến lược batch không chỉ tối ưu về mặt thời gian mà còn giảm thiểu fragmentation bộ nhớ do việc alloc/dealloc model liên tục, đồng thời tạo điều kiện cho GPU (nếu có) tận dụng tối đa băng thông tính toán thông qua kernel fusion và memory coalescing.
