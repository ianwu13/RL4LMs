import json
import os
import random
from tqdm import tqdm # type: ignore

class GamePlay:
    def __init__(self, config, agents) -> None:
        """Get the two model objects, maintain internal states."""
        self.config = config
        self.agents = agents

        self.setup_cxts()

    def game_play(self):
        """Make the two models interact with each other and store the logs."""
        
        all_convs = []
        for conv_ix in range(self.config.num_convs):
            print(f"Starting conv: {conv_ix}")

            conv = {
                "cxts": {},
                "utts": [],
                "results": {},
            }

            ag_cxts = self.choose_agent_contexts()
            # setup agent contexts - and reset internal storage.
            for ag, ag_cxt in zip(self.agents, ag_cxts):
                ag.reset_agent(ag_cxt)
                conv["cxts"][ag.name] = ag.cxt
            
            curr_ag = random.choice([0,1])
            for _ in tqdm(range(self.config.max_utts)):
                
                resp_obj = self.agents[curr_ag].respond()
                # check signature
                assert resp_obj["name"] == self.agents[curr_ag].name

                # save the response
                conv["utts"].append(resp_obj)

                # send the response to the partner
                self.agents[1 - curr_ag].receive(resp_obj)

                if self.conv_done(conv):
                    break

                curr_ag = 1 - curr_ag

            conv["results"] = self.get_conv_results(conv)

            all_convs.append(conv)

        print("All convs done; saving to a file.")
        out_path = os.path.join(self.config.results_dir, "convs.json")
        with open(out_path, "w") as f:
            json.dump(all_convs, f, indent=4)
        print(f"Convs stored at: {out_path}")

    def setup_cxts(self):
        """Prepare cxts from a dataset file."""
        rows = []
        with open(self.config.dataset_path, "r") as f:
            for ix, line in enumerate(f):
                if ix:
                    rows.append(line)

        all_cxt_pairs = set()
        for row in rows:

            items = row.split("<context>")
            assert len(items) == 3

            cxt1 = items[1].split("<history>")[0].strip()
            cxt2 = items[2].strip()

            cxt_pair = f"{cxt1}$$${cxt2}"
            all_cxt_pairs.add(cxt_pair)

        self.all_cxt_pairs = sorted(list(all_cxt_pairs))
        print(f"Extracted all cxt pairs from: {self.config.dataset_path}")
        print(f"Num unique cxt pairs: {len(self.all_cxt_pairs)}")
        
    def choose_agent_contexts(self):
        """Return a list of two randomly chosen contexts."""
        ag_cxts = random.choice(self.all_cxt_pairs).split("$$$")
        assert len(ag_cxts) == 2
        return ag_cxts

    def conv_done(self, conv):
        """Check if the conversation is done or not."""
        
        if conv["utts"][-1]["resp"] == "<selection>":
            return True
        
        return False

    def get_conv_results(self, conv):
        """Compute the results from a single conv.
        
        Per agent metrics
        avg # of words, points scored.

        Joint metrics
        Conv length, combined points, some measure of optimality (pareto optimal - may be just iterate over all possible divisions -- find the maximum joint points possible and compare how close is the joint points to it), indicator for finished or not.
        
        """
        return {}

    def save_overall_results(self):
        """Compute the results from all the convs and store to a file."""
        
        overall_results = {}

        out_path = os.path.join(self.config.results_dir, "overall_results.json")
        with open(out_path, "w") as f:
            json.dump(overall_results, f, indent=4)
        print(f"Overall results stored at: {out_path}")