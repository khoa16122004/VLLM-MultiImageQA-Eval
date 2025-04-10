import json
import os

# --- Tên file ---
file1_path = '../gt_image_paths.json'  # File chứa danh sách gốc (để kiểm tra sự tồn tại)
file2_path = '../test_mrag_bench_top5_modelencode=ReT_topkrerank=10.json'  # File chứa danh sách cần kiểm tra (đáp án)
output_txt_path = '../ReT_no-caption.txt' # File lưu kết quả các key có trùng khớp

# --- Hàm thực hiện so sánh và ghi file ---
def find_matching_keys(file1_path, file2_path, output_txt_path):
    """
    So sánh nội dung hai file JSON và ghi các key có trùng khớp ra file text.

    Args:
        file1_path (str): Đường dẫn đến file JSON thứ nhất.
        file2_path (str): Đường dẫn đến file JSON thứ hai.
        output_txt_path (str): Đường dẫn đến file text để ghi kết quả.
    """
    try:
        # 1. Đọc dữ liệu từ hai file JSON
        # print(f"Đang đọc dữ liệu từ '{file1_path}'...")
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        # print(f"Đọc '{file1_path}' thành công.")

        print(f"Đang đọc dữ liệu từ '{file2_path}'...")
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)
        # print(f"Đọc '{file2_path}' thành công.")

        # 2. Tìm các key có ít nhất một phần tử trùng khớp
        matching_keys = []
        print("Bắt đầu so sánh...")

        # Lặp qua các key và danh sách filename trong file 2 (file đáp án)
        for key, filenames_in_file2 in data2.items():
            # Kiểm tra xem key này có tồn tại trong file 1 không
            if key in data1:
                filenames_in_file1 = data1[key]
                # Chuyển danh sách filename trong file 1 thành set để tìm kiếm nhanh hơn O(1)
                filenames_set_file1 = set(filenames_in_file1)

                # Kiểm tra xem có bất kỳ filename nào trong file 2 tồn tại trong set của file 1 không
                found_match_for_key = False
                for filename in filenames_in_file2:
                    if filename in filenames_set_file1:
                        found_match_for_key = True
                        break # Chỉ cần tìm thấy 1 cái trùng là đủ cho key này

                # Nếu tìm thấy trùng khớp cho key này, thêm key vào danh sách kết quả
                if found_match_for_key:
                    matching_keys.append(key)
                    # print(f"  -> Tìm thấy trùng khớp cho key: '{key}'")
            else:
                pass
                # print(f"  - Cảnh báo: Key '{key}' tồn tại trong '{file2_path}' nhưng không có trong '{file1_path}'. Bỏ qua key này.")

        print(f"So sánh hoàn tất. Tìm thấy {len(matching_keys)} key có trùng khớp.")

        # 3. Ghi các key trùng khớp vào file text
        # print(f"Đang ghi kết quả vào '{output_txt_path}'...")
        with open(output_txt_path, 'w', encoding='utf-8') as f_out:
            if matching_keys:
                # Ghi mỗi key trên một dòng mới
                for key in matching_keys:
                    f_out.write(f"{key}\n")
                print(f"Ghi file thành công. Các key trùng khớp đã được lưu tại: {output_txt_path}")
            else:
                 f_out.write("") # Ghi file rỗng nếu không có key nào trùng
                 print("Không tìm thấy key nào có trùng khớp. File output sẽ rỗng.")


    # --- Xử lý lỗi ---
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn: {e.filename}")
    except json.JSONDecodeError as e:
        print(f"Lỗi: Định dạng JSON không hợp lệ trong file. Chi tiết: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")

# --- Chạy hàm chính ---
find_matching_keys(file1_path, file2_path, output_txt_path)