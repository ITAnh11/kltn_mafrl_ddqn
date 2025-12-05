import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# ⚙️ PHẦN CẤU HÌNH
# ==============================================================================
FILES_TO_COMPARE = {
    "MAFRL": "results/results_train_mafrl_ue_100_20251120_021325/mafrl_training_log.csv",
    "MAFRL (DDQN)": "results/results_train_mafrl_ddqn_ue_100_20251120_002322/mafrl_ddqn_training_log.csv",
    # "MAFRL": "results/results_train_mafrl_ue_20_20251121_014527/mafrl_training_log.csv",
    # "MAFRL (DDQN)": "results/results_train_mafrl_ddqn_ue_20_20251121_015449/mafrl_ddqn_training_log.csv",
}

WINDOW_SIZE = 200
# Thay vì 1 tên file, ta chỉ định thư mục lưu
OUTPUT_DIR = "results/results_train/"

# Tạo thư mục nếu chưa tồn tại để tránh lỗi
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==============================================================================


def load_and_label_data(filepath, policy_name):
    """Đọc file và gán nhãn policy."""
    try:
        if not os.path.exists(filepath):
            print(
                f"⚠️  CẢNH BÁO: Không tìm thấy file cho '{policy_name}' tại: {filepath}"
            )
            return None
        df = pd.read_csv(filepath)
        df["policy"] = policy_name
        print(f"✅ Đã tải '{policy_name}': {len(df)} dòng")
        return df
    except Exception as e:
        print(f"❌ Lỗi đọc file '{policy_name}': {e}")
        return None


# --- BƯỚC 1: TỰ ĐỘNG ĐỌC VÀ GỘP DỮ LIỆU (GIỮ NGUYÊN) ---
list_dfs = []
print("--- Bắt đầu tải dữ liệu ---")
for policy_name, file_path in FILES_TO_COMPARE.items():
    df = load_and_label_data(file_path, policy_name)
    if df is not None:
        list_dfs.append(df)

if not list_dfs:
    print("❌ Lỗi: Không có dữ liệu nào được tải thành công.")
    exit()

df_final = pd.concat(list_dfs, ignore_index=True)
print(f"--- Tổng cộng: {len(df_final)} dòng dữ liệu ---")


# --- BƯỚC 2: VẼ VÀ LƯU TỪNG HÌNH RIÊNG LẺ (ĐÃ SỬA) ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11})


def save_single_plot(data, x_col, y_col, title, ylabel, window, filename):
    """Hàm vẽ và lưu 1 biểu đồ duy nhất (MÀU TƯƠNG PHẢN + ĐƯỜNG ĐẬM)"""

    plt.figure(figsize=(10, 6))

    palette = {
        "MAFRL": "#1f77b4",  # Xanh dương (màu gốc mặc định)
        "MAFRL (DDQN)": "#ff7f0e",  # Cam (màu gốc mặc định)
    }

    # 1. Moving Average mượt hơn
    data[f"{y_col}_smooth"] = data.groupby("policy")[y_col].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

    # 2. Vẽ đường raw rất mờ (bóng nền)
    sns.lineplot(
        data=data,
        x=x_col,
        y=y_col,
        hue="policy",
        palette=palette,
        alpha=0.15,
        linewidth=0.8,
        legend=False,
    )

    # 3. Vẽ đường smooth chính (ĐẬM + RÕ MÀU)
    sns.lineplot(
        data=data,
        x=x_col,
        y=f"{y_col}_smooth",
        hue="policy",
        palette=palette,
        linewidth=2.5,
    )

    plt.title(title, fontweight="bold")
    plt.ylabel(ylabel)
    plt.xlabel("Episode")

    # ✅ GIỚI HẠN TRỤC X TỐI ĐA 30000 EPISODES
    plt.xlim(0, 30000)

    plt.tight_layout()

    # Lưu file
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Đã lưu ảnh: {full_path}")

    plt.close()


# --- THỰC HIỆN VẼ TỪNG CÁI ---

# 1. Vẽ Reward
save_single_plot(
    df_final,
    "episode",
    "reward",
    "Reward Convergence",
    "Reward",
    WINDOW_SIZE,
    "reward_comparison.png",
)

# 2. Vẽ Power
save_single_plot(
    df_final,
    "episode",
    "total_power_system",
    "System Power Consumption",
    "Power (W)",
    WINDOW_SIZE,
    "power_comparison.png",
)

# 3. Vẽ Violations
save_single_plot(
    df_final,
    "episode",
    "violations",
    "SLA Violations",
    "Count",
    WINDOW_SIZE,
    "violations_comparison.png",
)

print("🎉 Hoàn tất!")
