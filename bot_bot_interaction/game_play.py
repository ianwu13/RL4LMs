import json
import os
import random
from tqdm import tqdm

from bot_bot_interaction.predict_agreed_deal import PredictAgreedDeal

class GamePlay:
    def __init__(self, config, agents) -> None:
        """Get the two model objects, maintain internal states."""
        self.config = config
        self.agents = agents

        self.setup_cxts()

        self.predict_deal_obj = PredictAgreedDeal(self.config)

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

    def get_pref_values(self, cxt):
        """Get issue-wise preference values from the given cxt."""
        items = cxt.strip().split()
        assert len(items) == 11

        i1_c, i2_c, i3_c = int(items[3]), int(items[6].rstrip(",")), int(items[9])
        pps = [i1_c, i2_c, i3_c]

        issues = None
        if "book" in cxt:
            issues = ["book", "hat", "ball"]
        else:
            issues = ["food", "water", "firewood"]
        
        prefs = {}

        for iss, pp in zip(issues, pps):
            prefs[iss] = pp

        return prefs

    def compute_points_scored(self, prefs, deal, mname):
        """Compute the points scored."""

        points = 0
        for issue, p in prefs.items():
            points += deal[mname][issue]*p

        return points

    def get_feasible_deals(self, conv):
        """Get a list of all feasible deals."""
        
        # first get the counts of the three items in this dialogue.
        cxt = list(conv["cxts"].values())[0]
        items = cxt.strip().split()
        assert len(items) == 11

        i1_c, i2_c, i3_c = int(items[2].rstrip(",")), int(items[5].rstrip(",")), int(items[8].rstrip(","))
        ccs = [i1_c, i2_c, i3_c]

        issues = None
        if "book" in cxt:
            issues = ["book", "hat", "ball"]
        else:
            issues = ["food", "water", "firewood"]
        
        counts = {}
        for iss, cc in zip(issues, ccs):
            counts[iss] = cc

        mnames = list(conv["cxts"].keys())
        all_deals = []

        for d1 in range(counts[issues[0]] + 1):
            for d2 in range(counts[issues[1]] + 1):
                for d3 in range(counts[issues[2]] + 1):
                    new_deal = {
                        mnames[0]: {
                            issues[0]: d1,
                            issues[1]: d2,
                            issues[2]: d3,
                        },
                        mnames[1]: {
                            issues[0]: counts[issues[0]] - d1,
                            issues[1]: counts[issues[1]] - d2,
                            issues[2]: counts[issues[2]] - d3,
                        }
                    }
                    all_deals.append(new_deal)
        
        return all_deals

    def get_deal_points(self, conv, deal):
        """Get the points scored by both players according to the specific deal."""
        
        deal_s = {}

        for mname in conv["cxts"].keys():
            # get issue-wise pref values
            prefs = self.get_pref_values(conv["cxts"][mname])

            # get the points scored
            points = self.compute_points_scored(prefs, deal, mname)

            deal_s[mname] = points

        return deal_s

    def check_pareto_optimal(self, conv, deal):
        """Check if the agreed deal is pareto optimal.
        
        A deal is pareto optimal if the score of one cannot be improved any further without lowering the partner's score."""
        mnames = list(conv["cxts"].keys())

        feasible_deals = self.get_feasible_deals(conv)

        deal_s = self.get_deal_points(conv, deal)
        s1, s2 = deal_s[mnames[0]], deal_s[mnames[1]]

        can_improve = False
        for cand_deal in feasible_deals:
            cand_deal_s = self.get_deal_points(conv, cand_deal)
            cs1, cs2 = cand_deal_s[mnames[0]], cand_deal_s[mnames[1]]

            if (cs1 > s1 and cs2 >= s2) or (cs2 > s2 and cs1 >= s1):
                can_improve = True
                break

        if can_improve:
            # i.e. not optimal.
            return False
        
        return True

    def get_conv_results(self, conv):
        """Compute the results from a single conv.
        
        Per agent metrics
        avg # of words, points scored.

        Joint metrics
        Conv length, whether agreed deal was detected, combined points, pareto_optimal, indicator for finished or not.
        
            conv = {
                "cxts": {},
                "utts": [],
                "results": {},
            }
        """

        results = {
            "per_model": {},
            "joint": {},
        }

        if conv["utts"][-1]["resp"] == "<selection>":
            results["joint"]["conv_finished"] = 1
        else:
            results["joint"]["conv_finished"] = 0
            return results

        # for each mname - get the issue-wise counts for each issue.
        deal = self.predict_deal_obj.get_deal(conv)

        # compute per model metrics.
        for mname in conv["cxts"].keys():
            
            results["per_model"][mname] = {}

            #no of words
            num_words = []
            for utt in conv["utts"]:
                if conv["name"] == mname:
                    num_words.append(len(utt["resp"].split()))
            
            results["per_model"][mname]["num_words"] = sum(num_words) / len(num_words)

            if deal:

                # get issue-wise pref values
                prefs = self.get_pref_values(conv["cxts"][mname])

                # get the points scored
                points = self.compute_points_scored(prefs, deal, mname)

                # save
                results["per_model"][mname]["points"]  = points
        
        # compute joint metrics
        results["joint"]["num_utts"] = len(conv["utts"])

        if deal:
            results["joint"]["deal_detected"] = 1

            joint_points = 0
            for mname in conv["cxts"].keys():
                joint_points += results["per_model"][mname]["points"]
            results["joint"]["joint_points"] = joint_points
            
            # just like here: https://github.com/facebookresearch/end-to-end-negotiator/blob/bbb93bbf00f69fced75d5c0d22e855bda07c9b78/src/eval_selfplay.py#L117
            is_pareto_optimal = self.check_pareto_optimal(conv, deal)
            
            results["joint"]["pareto_optimal"] = int(is_pareto_optimal)
        else:
            results["joint"]["deal_detected"] = 0

        return joint_points

    def save_overall_results(self):
        """Compute the results from all the convs and store to a file."""
        
        overall_results = {}

        out_path = os.path.join(self.config.results_dir, "overall_results.json")
        with open(out_path, "w") as f:
            json.dump(overall_results, f, indent=4)
        print(f"Overall results stored at: {out_path}")