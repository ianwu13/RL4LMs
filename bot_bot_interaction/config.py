import os
from agents import SupervisedAgent, OfflineRLAgent

class Config:
    def __init__(self) -> None:

        # I/O
        self.log_dir = "/home/ICT2000/chawla/nego_rl/logs/rl4lm_exps/"
        self.out_dir = "/home/ICT2000/chawla/nego_rl/logs/bot_bot_results/"
        self.model_names = [
            "comb_perturbs_dt_1_bs16",
            "comb_no_rl_dt_2",
        ]
        self.model_names = [f"{item}_ix{ix}" for ix, item in enumerate(self.model_names)]
        self.model_types = ["offline_rl", "supervised"]
        self.override_results = True

        self.model_typ2class = {
            "offline_rl": OfflineRLAgent,
            "supervised": SupervisedAgent,
        }
        
        # to get the initial agent contexts
        self.dataset_path = "/home/ICT2000/chawla/nego_rl/data/offline_rl_dt/dealornodeal/eval.csv"

        # interaction
        self.num_convs = 3 # total conversations to be logged.
        self.max_utts = 20 # max utterances in one conversation - hard stop.

        # process
        self.results_dir = os.path.join(self.out_dir, f"{self.model_names[0]}_{self.model_names[1]}", "")