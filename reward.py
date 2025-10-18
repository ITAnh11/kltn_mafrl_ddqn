import pandas as pd
import matplotlib.pyplot as plt

algos = ["MAFRL_DDQN", "MARL", "MAFRL"]
window = 500  # moving average window, có thể chỉnh để mượt hơn

plt.figure(figsize=(10, 5))

for algo in algos:
    df = pd.read_csv(f"results/{algo}_reward.csv")
    df["Reward_MA"] = df["Reward"].rolling(window).mean()
    plt.plot(df["Episode"], df["Reward_MA"], linewidth=2, label=algo)

plt.xlabel("Episode")
plt.ylabel("Reward (Moving Average)")
plt.title(f"Reward Moving Average Comparison (window={window})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("reward_comparison_ma.png", dpi=300)
plt.show()
