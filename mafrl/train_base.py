import os
from mafrl.mafrl_base import MAFRL
from datetime import datetime

num_ue = 20  # số UE cố định để train MARL

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"results/results_train_mafrl_new_ue_{num_ue}_{timestamp}"
os.makedirs(outdir, exist_ok=True)


mafrl = MAFRL(num_ues=num_ue, num_uavs=10, num_episodes=40000, outdir=outdir)
time_start = datetime.now()
mafrl.train()
mafrl.save_model(f"mafrl/model/mafrl_model_new_ue_{num_ue}_{timestamp}.pth")
time_end = datetime.now()
print(f"Training started at: {time_start}")
print(f"Training ended at: {time_end}")
print(f"Total training time: {time_end - time_start}")
# save the training time to a text file
with open(os.path.join(outdir, "training_time_mafrl_base_new.txt"), "w") as f:
    f.write(f"Training started at: {time_start}\n")
    f.write(f"Training ended at: {time_end}\n")
    f.write(f"Total training time: {time_end - time_start}\n")
