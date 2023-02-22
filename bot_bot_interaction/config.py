import os
from agents import SupervisedAgent, OfflineRLAgent

class Config:
    def __init__(self) -> None:
        
        # Step 1: Give the names of the model directories under the log folder.
        self.model_names = [
            "nds_dnd1",
            "nds_dnd1",
        ]

        # Step 2: Give the types of these two models in the same order, from ["supervised", "offline_rl"]
        self.model_types = ["supervised", "supervised"]
        assert len(self.model_types) == len(self.model_names) == 2

        # Step 3: Give the path to the dataset for extracting initial agent contexts
        self.dataset_path = "/home/ICT2000/chawla/nego_rl/data/nego_dialog_sel_offline_rl_dt/dealornodeal/validation.csv"

        # Step 4: Define interaction params
        self.num_convs = 100 # total conversations to be logged.
        self.max_utts = 20 # max utterances in one conversation - hard stop.
        assert not self.num_convs % 2
        
        # I/O - mostly fixed from here on.
        self.log_dir = "/home/ICT2000/chawla/nego_rl/logs/rl4lm_exps/"
        self.out_dir = "/home/ICT2000/chawla/nego_rl/logs/bot_bot_results/"
        
        self.model_names = [f"{item}_ix{ix}" for ix, item in enumerate(self.model_names)]
        
        self.override_results = True

        self.model_typ2class = {
            "offline_rl": OfflineRLAgent,
            "supervised": SupervisedAgent,
        }

        # process
        self.results_dir = os.path.join(self.out_dir, f"{self.model_names[0]}_{self.model_names[1]}")

        # predict agreed deal model
        self.predict_deal_model_path = "/home/ICT2000/chawla/nego_rl/logs/rl4lm_exps/predict_deal_comb1/model/"