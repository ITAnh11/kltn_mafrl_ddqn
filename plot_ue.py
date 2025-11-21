import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Cấu hình đường dẫn
csv_path = "results/results_ue/results_ue.csv"  # Đường dẫn file CSV
output_folder = "results/results_ue/plots"  # Thư mục để lưu ảnh đầu ra

# Tạo thư mục lưu ảnh nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Đọc dữ liệu
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Không tìm thấy file tại: {csv_path}")
    # Tạo dữ liệu giả lập nếu không tìm thấy file (để test code)
    import io

    csv_data_dummy = """num_ues,policy,reward,total_power_system,total_power_ue,total_power_uav,violations,vio_local,vio_offload
    20,Greedy Offload,0.04,2.0,2.0,0.001,1.0,0.0,1.0
    20,Random Offload,0.05,1.1,1.1,0.001,1.5,0.7,0.8
    20,Local Execution,0.02,0.1,0.1,0.0,8.2,8.2,0.0
    """
    df = pd.read_csv(io.StringIO(csv_data_dummy))

# 2. Cấu hình giao diện chung
sns.set_theme(style="whitegrid")
metrics = [
    "reward",
    "total_power_system",
    "total_power_ue",
    "total_power_uav",
    "violations",
    "vio_local",
    "vio_offload",
]

# 3. Vòng lặp vẽ và lưu từng biểu đồ
print("Đang bắt đầu vẽ và lưu biểu đồ...")

for metric in metrics:
    # Tạo một figure mới cho mỗi metric
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="num_ues",
        y=metric,
        hue="policy",
        style="policy",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="tab10",  # Dùng bảng màu chuẩn để dễ nhìn
    )

    # Trang trí biểu đồ
    plt.title(f"Comparison of {metric}", fontsize=14, fontweight="bold")
    plt.xlabel("Number of UEs", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(title="Policy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Tự động căn chỉnh lề
    plt.tight_layout()

    # Xây dựng tên file và lưu
    # Ví dụ: plots/chart_reward.png
    file_name = f"chart_{metric}.png"
    save_path = os.path.join(output_folder, file_name)

    plt.savefig(save_path, dpi=300)  # dpi=300 để ảnh sắc nét chèn vào báo cáo
    print(f"Đã lưu: {save_path}")

    # Đóng figure để giải phóng bộ nhớ (quan trọng khi chạy vòng lặp)
    plt.close()
