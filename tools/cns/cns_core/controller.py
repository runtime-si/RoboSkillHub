# ruff: noqa  (vendored from CNS @ 6b42138, MIT)
# From CNS (https://github.com/hhcaz/CNS @ 6b42138), cns/benchmark/controller.py, MIT, Copyright (c) 2023 hhcaz.
# Changed: only GraphVSController; the "new scene" print removed; the checkpoint is loaded with weights_only.
import numpy as np
import torch

from .midend.graph_gen import GraphData
from .models.graph_vs import GraphVS


class GraphVSController:
    def __init__(self, ckpt_path: str, device="cuda:0"):
        self.device = torch.device(device)
        # self.net: GraphVS = torch.load(ckpt_path, map_location=self.device, weights_only=True)["net"]
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        if hasattr(ckpt, "net") and isinstance(ckpt["net"], torch.nn.Module):
            self.net: GraphVS = ckpt["net"]
        else:
            self.net = GraphVS(2, 2, 128, regress_norm=True).to(device)
            self.net.load_state_dict(ckpt)
        self.net.eval()
        self.hidden = None

    def __call__(self, data: GraphData) -> np.ndarray:
        with torch.no_grad():
            data = data.to(self.device)
            if hasattr(self.net, "preprocess"):
                data = self.net.preprocess(data)

            if data.new_scene.any():
                self.hidden = None

            raw_pred = self.net(data, self.hidden)
            self.hidden = raw_pred[-1]
            vel = self.net.postprocess(raw_pred, data)

        vel = vel.squeeze(0).cpu().numpy()
        return vel
