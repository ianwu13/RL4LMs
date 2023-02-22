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

        self.predict_deal_obj = PredictAgreedDeal(self.config)

        # set up the contexts
        self.cxt_pairs_path = self.config.dataset_path.replace(".csv", "_cxt_pairs.json")
        if not os.path.exists(self.cxt_pairs_path):
            self.setup_cxts()
        # now load the required number of cxt pairs from the top of the file.
        self.load_cxts()

        self.all_convs = None

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

            # choose the two contexts for this conv
            ag_cxts = self.chosen_cxt_pairs[conv_ix].split("$$$")

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

        self.all_convs = all_convs[:]

    def setup_cxts(self):
        """Prepare cxts from a dataset file - remove duplicates, shuffle and to persist the order, store them to a file.
        """

        rows = []
        with open(self.config.dataset_path, "r") as f:
            for ix, line in enumerate(f):
                if ix:
                    rows.append(line)

        cxt_pair_set = set()

        for row in rows:

            items = row.split("<context>")
            assert len(items) == 3

            cxt1 = items[1].split("<history>")[0].strip()
            cxt2 = items[2].replace("\"","").strip()

            opt1, opt2 = f"{cxt1}$$${cxt2}", f"{cxt2}$$${cxt1}"

            if opt1 in cxt_pair_set or opt2 in cxt_pair_set:
                continue

            # otherwise
            cxt_pair_set.add(opt1)

        for item in cxt_pair_set:
            reverse_item = f"{item.split('$$$')[1]}$$${item.split('$$$')[0]}"
            if reverse_item != item:
                assert reverse_item not in cxt_pair_set, f"{item} --- {reverse_item}"

        cxt_pairs = sorted(list(cxt_pair_set))
        random.shuffle(cxt_pairs)

        # save them to a file.
        with open(self.cxt_pairs_path, "w") as f:
            json.dump({"cxts": cxt_pairs}, f)
        print(f"cxt pairs stored at: {self.cxt_pairs_path}")

    def load_cxts(self):
        """Load the required number of cxt pairs from a file.

        Basically - now we extract unique pairs from the file -> and then make sure both (cxt1, cx2) and (cxt2, cxt1) are covered. That is why we will always simulate an even no. of conversations in total.
        """

        with open(self.cxt_pairs_path, "r") as f:
            cxt_pairs = json.load(f)["cxts"][: self.config.num_convs // 2]
        
        final_cxt_pairs = []
        for cxt_pair in cxt_pairs:
            
            rev_pair = f"{cxt_pair.split('$$$')[1]}$$${cxt_pair.split('$$$')[0]}"

            # add both - regardless of whether they are the same.
            final_cxt_pairs += [cxt_pair, rev_pair]
        
        assert len(final_cxt_pairs) == self.config.num_convs
        
        self.chosen_cxt_pairs = final_cxt_pairs[:]
        print("Required cxt pairs loaded from file.")

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

        per_model:
        - num_words
        - points

        joint:
        - conv_finished
        - deal_detected
        - num_utts
        - joint_points
        - pareto_optimal

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
        """Compute the results from all the convs and store to a file.
        
        per_model:
        - num_words - avg for conv finished.
        - points - avg for the cases where conv was finished and deal was detected.

        joint:
        - conv_finished - fract
        - deal_detected - fract. for conv finished
        - num_utts - avg for conv finished
        - joint_points - avg for conv finished and deal detected.
        - pareto_optimal - frac for conv finished and deal detected.
        """
        assert self.all_convs

        overall_results = {
            "per_model": {},
            "joint": {},
        }

        mnames = list(self.all_convs[0]["results"]["per_model"].keys())

        for mname in mnames:
            overall_results["per_model"][mname] = {}
        
        # per_model
        for mname in mnames:
            num_words, points = [], []
            for conv in self.all_convs:
                if conv["results"]["joint"]["conv_finished"]:
                    num_words.append(conv["results"]["per_model"][mname]["num_words"])
                
                    if conv["results"]["joint"]["deal_detected"]:
                        points.append(conv["results"]["per_model"][mname]["points"])
            
            if num_words:
                overall_results["per_model"][mname]["num_words"] = sum(num_words) / len(num_words)
            
            if points:
                overall_results["per_model"][mname]["points"] = sum(points) / len(points)
        
        # joint
        conv_finished, deal_detected, num_utts, joint_points, pareto_optimal = [], [], [], [], []
        for conv in self.all_convs:
            conv_finished.append(conv["results"]["joint"]["conv_finished"])

            if conv["results"]["joint"]["conv_finished"]:
                deal_detected.append(conv["results"]["joint"]["deal_detected"])

                num_utts.append(conv["results"]["joint"]["num_utts"])

                if conv["results"]["joint"]["deal_detected"]:
                    joint_points.append(conv["results"]["joint"]["joint_points"])

                    pareto_optimal.append(conv["results"]["joint"]["pareto_optimal"])

        if conv_finished:
            overall_results["joint"]["conv_finished"] = sum(conv_finished) / len(conv_finished)

        if deal_detected:
            overall_results["joint"]["deal_detected"] = sum(deal_detected) / len(deal_detected)

        if num_utts:
            overall_results["joint"]["num_utts"] = sum(num_utts) / len(num_utts)

        if joint_points:
            overall_results["joint"]["joint_points"] = sum(joint_points) / len(joint_points)

        if pareto_optimal:
            overall_results["joint"]["pareto_optimal"] = sum(pareto_optimal) / len(pareto_optimal)

        out_path = os.path.join(self.config.results_dir, "overall_results.json")
        with open(out_path, "w") as f:
            json.dump(overall_results, f, indent=4)
        print(f"Overall results stored at: {out_path}")