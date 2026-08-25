[Skip to content](https://cord-babcat-65f.notion.site/MEETING-HCMAI-3bb2b1c123f680cabd13ec6d4fca24d1#main)

# MEETING HCMAI

#### Luồng hiện tại

Video → shot detection → lấy mẫu/keyframe → embedding → Qdrant → tìm trên collection clip và keyframe → hợp nhất/rerank bằng RRF → Top-K → Kết quả

Phase 1: Hoàn thiện baseline

Phase 2: Agent

#### 1\. Tổng hợp và phân công

\[ \] Thành tổng hợp lại các công việc

\[ \] Tùng duyệt lại và gửi bản phân công

#### 2\. Thiết lập môi trường & Chạy thử hệ thống

\[ \] Chuẩn bị phần cứng: Đảm bảo máy chạy thử nghiệm có GPU với tối thiểu 4GB VRAM

\[ \] Chạy Docker & Cơ sở dữ liệu:

Dựng cơ sở dữ liệu vector Qdrant bằng Docker Compose.

Sử dụng file snapshot Tùng gửi để import (hiện mới có khoảng 10 video test với hơn 5.300 vectors).

Verify hoạt động của Qdrant qua cổng

localhost:6333/dashboard

.

\[ \] Khởi chạy source code:

Clone 2 repo Frontend và Backend về máy.

Cài đặt Node.js, cài thư viện (

npm i

) và chạy thử cả Frontend (

npm run dev

) lẫn Backend.

(Cân nhắc): Gộp chung 2 repo này lại thành một để cả nhóm dễ quản lý và theo dõi code nếu cần.

#### 3\. Cải tiến mô hình tìm kiếm & Đánh giá

\[ \] Xây dựng tính năng Ablation Study: Thiết kế hệ thống cho phép bật/tắt linh hoạt từng thành phần tìm kiếm (chỉ dùng SigLIP, chỉ dùng Clip, chỉ dùng OCR, chỉ dùng ASR, hoặc kết hợp...) để đánh giá chính xác vai trò và hiệu quả đóng góp của từng module đối với kết quả cuối cùng.

\[ \] Nâng cấp mô hình CLIP: Thử nghiệm thay thế model cũ bằng Jina-Clip (phiên bản V2) để cải thiện khả năng biểu diễn không gian ngữ nghĩa giữa văn bản và hình ảnh.

\[ \] Thử nghiệm các mô hình VLM khác: Tìm hiểu và đánh giá thêm các mô hình Vision-Language mạnh hơn như InternVL hoặc các mô hình thuộc họ Win-VL hỗ trợ tìm kiếm đa phương thức (multimodal).

#### 4\. Bổ sung các nhánh trích xuất Metadata (OCR, ASR, YOLO, Captioning)

\[ \] Tận dụng ASR (Speech-to-Text): Tải phần transcript (phụ đề/giọng nói) từ các link YouTube do ban tổ chức cung cấp để làm giàu dữ liệu văn bản đi kèm video, giúp tìm kiếm các phân cảnh có lời thoại.

\[ \] Tối ưu hóa OCR: Trích xuất ký tự trong các khung cảnh (slide bài giảng, biển hiệu, đồng hồ đếm ngược, bảng số...) để phục vụ cho các câu truy vấn dạng văn bản hiển thị trên màn hình.

\[ \] Tích hợp Object Detection (YOLO): Sử dụng các mô hình YOLO (như YOLOv8, YOLO-World hoặc YOLO-NAS) để phát hiện và đếm số lượng vật thể trong keyframe (ví dụ: đếm số người, số xe hơi...) phục vụ cho việc lọc (filter) metadata.

\[ \] Sinh Caption tự động (VLM): Sử dụng mô hình VLM để viết mô tả chi tiết cho từng keyframe. Thay vì chỉ dùng CLIP, việc kết hợp caption chi tiết bằng văn bản sẽ giúp tìm kiếm ngữ nghĩa chính xác hơn.

#### 5\. Cải tiến cơ chế xử lý Query (Query Processing & Reranking)

\[ \] Viết lại truy vấn (Query Rewriting): Sử dụng LLM để dịch các câu truy vấn tiếng Việt sang tiếng Anh, đồng thời mở rộng câu truy vấn gốc thành 4-5 câu có cùng ngữ nghĩa nhưng sử dụng từ vựng khác nhau.

\[ \] Truy vấn đa luồng & Rerank: Gửi các câu truy vấn đã mở rộng vào hệ thống để tìm kiếm song song trên các collection của Qdrant, sau đó tổng hợp và sắp xếp lại kết quả bằng thuật toán Rerank (ví dụ: RRF - Reciprocal Rank Fusion).

#### 6\. Chuẩn bị tập dữ liệu đánh giá (Ground Truth & Testing)

\[ \] Dán nhãn dữ liệu kiểm thử (Ground Truth): Chuẩn bị sẵn một danh sách các câu hỏi mẫu kèm theo đáp án chính xác (Video ID, số thứ tự Frame/Shot) đã được xác nhận thủ công.

\[ \] Đóng băng code (Code Freeze): Dự kiến dừng phát triển tính năng mới vào ngày 18 (khoảng 2-3 ngày trước ngày thi chính thức)