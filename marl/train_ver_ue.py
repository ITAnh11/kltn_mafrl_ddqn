import os
from marl.marl import MARL
from datetime import datetime

num_ue = 100  # số UE cố định để train MARL

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"results/results_train_marl_ue_{num_ue}_{timestamp}"
os.makedirs(outdir, exist_ok=True)


marl = MARL(num_ues=num_ue, num_uavs=10, num_episodes=30000, outdir=outdir)
time_start = datetime.now()
marl.train()
marl.save_model(f"marl/model/marl_model_ue_{num_ue}_{timestamp}.pth")
time_end = datetime.now()
print(f"Training started at: {time_start}")
print(f"Training ended at: {time_end}")
print(f"Total training time: {time_end - time_start}")
# save the training time to a text file
with open(os.path.join(outdir, "training_time_mafrl_base_new.txt"), "w") as f:
    f.write(f"Training started at: {time_start}\n")
    f.write(f"Training ended at: {time_end}\n")
    f.write(f"Total training time: {time_end - time_start}\n")
