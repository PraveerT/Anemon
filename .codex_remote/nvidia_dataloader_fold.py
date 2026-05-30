"""Fold-aware point loader for honest out-of-fold (OOF) calibration.

Holds out subjects listed in $HOLDOUT_SUBJECTS (comma-sep, e.g. 'subject11_,
subject15_,subject18_') from the train phase. With $HOLDOUT_AS_TEST=1 the 'test'
phase returns those held-out train clips instead of the real test set, so the
trainer dumps clean OOF logits for the held-out subjects (models never trained
on them). Env vars unset -> identical to the base loader.
"""
import os
from nvidia_dataloader import NvidiaLoader


class NvidiaLoaderFold(NvidiaLoader):
    def get_inputs_list(self):
        prefix = "../dataset/Nvidia/Processed"
        H = [s for s in os.environ.get("HOLDOUT_SUBJECTS", "").split(",") if s]
        trainf = prefix + ("/train_depth_list.txt" if self.datatype == "depth" else "/train_color_list.txt")
        testf = prefix + ("/test_depth_list.txt" if self.datatype == "depth" else "/test_color_list.txt")
        if self.phase == "train":
            lines = open(trainf).readlines()
            lines = [l for l in lines if ("subject" + str(self.valid_subject) + "_") not in l]
            if H:
                lines = [l for l in lines if not any(h in l for h in H)]
            return lines
        if self.phase == "valid":
            lines = open(trainf).readlines()
            return [l for l in lines if ("subject" + str(self.valid_subject) + "_") in l]
        if self.phase == "test":
            if os.environ.get("HOLDOUT_AS_TEST") and H:
                lines = open(trainf).readlines()
                return [l for l in lines if any(h in l for h in H)]
            return open(testf).readlines()
        raise AssertionError("Phase error.")
