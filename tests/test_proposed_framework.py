import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm.config import load_experiment
from alm.data.audibility import (dominant_mask_share, resize_masks,
                                 source_time_frequency_masks, source_to_rest_ratio_db,
                                 stft_power)
from alm.data.labels import audibility_score, construct_labels, union_labels
from alm.losses.asymmetric import AsymmetricLoss
from alm.losses.combined import CombinedLoss, mask_cross_entropy
from alm.losses.cost_sensitive import (estimate_training_cooccurrence,
                                       negative_weights_from_matrix,
                                       validate_cost_matrix)
from alm.models.alm_model import ALMModel
from alm.training.engine import Trainer
from alm.training.metrics import calibrate_thresholds, multilabel_metrics


class AudibilityTests(unittest.TestCase):
    def test_source_to_rest_and_silence(self):
        source = np.ones(32)
        mixture = source + np.ones(32)
        self.assertAlmostEqual(source_to_rest_ratio_db(source, mixture), 0.0, places=5)
        self.assertEqual(source_to_rest_ratio_db(np.zeros(32), mixture), float("-inf"))
        self.assertEqual(source_to_rest_ratio_db(source, source), float("inf"))

    def test_masks_are_finite_aligned_and_support_overlap(self):
        contributions = np.stack([np.ones(1600), np.ones(1600)])
        masks = source_time_frequency_masks(contributions, sample_rate=16000, n_fft=128,
                                            hop_length=64, n_mels=16)
        self.assertEqual(masks.shape[0], 2)
        self.assertTrue(np.isfinite(masks).all())
        self.assertTrue(np.all((masks >= 0) & (masks <= 1)))
        self.assertTrue(np.allclose(masks.sum(0)[masks.sum(0) > 0], 1.0, atol=1e-5))
        resized = resize_masks(masks, 8, 10)
        self.assertEqual(resized.shape, (2, 8, 10))
        self.assertGreaterEqual(dominant_mask_share(masks, source_index=0), 0.0)
        self.assertEqual(stft_power(np.ones(1600), n_fft=128, hop_length=64).shape[0], 65)


class LabelTests(unittest.TestCase):
    def test_policies_and_soft_floor(self):
        self.assertTrue(np.array_equal(union_labels([1, 4]), [0, 1, 0, 0, 1, 0, 0]))
        threshold = construct_labels([1, 4], policy="audibility_threshold", audibility_db=[-20, 2])
        self.assertTrue(np.array_equal(threshold, [0, 0, 0, 0, 1, 0, 0]))
        soft = construct_labels([1], policy="soft_audibility", audibility_db=[-100], soft_floor=0.1)
        self.assertGreater(float(soft[1]), 0.0)
        self.assertLess(float(soft[1]), 0.2)
        self.assertEqual(audibility_score(float("inf")), 1.0)
        self.assertEqual(audibility_score(float("-inf")), 0.0)


class LossTests(unittest.TestCase):
    def test_bce_matches_torch_and_asl_gradients_are_finite(self):
        logits = torch.randn(3, 7, requires_grad=True)
        targets = torch.rand(3, 7)
        pos_weight = torch.rand(7) + 0.5
        ours = CombinedLoss(pos_weight=pos_weight)(logits, targets)["total"]
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        self.assertTrue(torch.allclose(ours, expected))
        loss = AsymmetricLoss()(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mask_loss_excludes_absent_classes(self):
        attention = torch.full((2, 7, 4, 5), 1 / 20)
        target = torch.zeros_like(attention)
        target[0, 0, 1, 2] = 1
        target[1, 2, 2, 3] = 1
        valid = torch.ones(2, 4, 5, dtype=torch.bool)
        result = mask_cross_entropy(attention, target, valid_mask=valid,
                                    class_valid=torch.tensor([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]], dtype=torch.bool))
        self.assertTrue(torch.isfinite(result))

    def test_cost_matrix_validation(self):
        matrix = torch.zeros(7, 7)
        matrix[0, 1] = 2
        self.assertEqual(tuple(validate_cost_matrix(matrix).shape), (7, 7))
        weights = negative_weights_from_matrix(torch.tensor([[1., 1., 0, 0, 0, 0, 0]]), matrix)
        self.assertAlmostEqual(float(weights[0, 0]), 3.0)
        estimated = estimate_training_cooccurrence(torch.tensor([[1., 1., 0, 0, 0, 0, 0], [1., 0, 1, 0, 0, 0, 0]]))
        self.assertTrue(torch.isfinite(estimated).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the device-placement regression test")
    def test_cost_sensitive_loss_uses_one_device(self):
        device = torch.device("cuda")
        matrix = torch.zeros(7, 7, device=device)
        matrix[0, 1] = 1.0
        loss_fn = CombinedLoss(pos_weight=torch.ones(7), confusion_matrix=matrix).to(device)
        logits = torch.zeros(2, 7, device=device, requires_grad=True)
        targets = torch.zeros(2, 7, device=device)
        value = loss_fn(logits, targets)["total"]
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(logits.grad).all())


class ModelTests(unittest.TestCase):
    def test_b0_and_full_forward_contracts(self):
        mel = torch.randn(2, 1, 128, 32)
        b0 = ALMModel(embed_dim=32, num_heads=4, num_layers=1, ff_dim=64,
                      max_time_frames=10, use_fusion=False, use_class_queries=False).eval()
        full = ALMModel(embed_dim=32, num_heads=4, num_layers=1, ff_dim=64,
                        max_time_frames=10, use_fusion=True, use_class_queries=True).eval()
        with torch.no_grad():
            self.assertEqual(tuple(b0(mel).shape), (2, 7))
            output = full(mel, return_aux=True)
        self.assertEqual(tuple(output["logits"].shape), (2, 7))
        self.assertEqual(tuple(output["attention"].shape), (2, 7, 8, 8))
        self.assertTrue(torch.allclose(output["attention"].flatten(2).sum(-1), torch.ones(2, 7), atol=1e-5))

    def test_full_objective_has_finite_gradients(self):
        model = ALMModel(embed_dim=32, num_heads=4, num_layers=1, ff_dim=64,
                         max_time_frames=10, use_fusion=True, use_class_queries=True)
        mel = torch.randn(2, 1, 128, 32)
        target = torch.rand(2, 7)
        audibility = torch.where(target > 0.5, torch.zeros_like(target), torch.full_like(target, float("-inf")))
        output = model(mel, return_aux=True)
        masks = torch.rand_like(output["attention"])
        loss = CombinedLoss(pos_weight=torch.ones(7), use_asymmetric_loss=True,
                            use_audibility_loss=True, lambda_mask=0.25)(
                                output["logits"], target, audibility_db=audibility,
                                attention_maps=output["attention"], mask_targets=masks,
                                class_valid=target > 0.5)["total"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


class InfrastructureTests(unittest.TestCase):
    def test_thresholds_and_smoke_training(self):
        y = np.array([[1, 0, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0]], dtype=np.int32)
        scores = y * 0.8 + (1 - y) * 0.2
        calibration = calibrate_thresholds(y, scores)
        metrics = multilabel_metrics(y, scores, thresholds=calibration["thresholds"])
        self.assertIn("mAP", metrics)
        with tempfile.TemporaryDirectory(dir="/tmp/omnirush") as directory:
            root = Path(directory)
            model = ALMModel(embed_dim=32, num_heads=4, num_layers=1, ff_dim=64,
                             max_time_frames=10, use_fusion=False, use_class_queries=False)
            inputs = torch.randn(4, 1, 128, 32)
            targets = torch.randint(0, 2, (4, 7)).float()
            loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
            # Adapt the two-tensor fixture to the engine's documented batch keys.
            class Adapter:
                def __iter__(self):
                    for x, y in loader:
                        yield {"mel": x, "target": y, "union_target": y}
                def __len__(self): return len(loader)
            trainer = Trainer(model, CombinedLoss(pos_weight=torch.ones(7)), torch.optim.AdamW(model.parameters(), lr=1e-3),
                              device=torch.device("cpu"), run_dir=root / "run", epochs=1, amp=False)
            result = trainer.fit(Adapter(), Adapter())
            self.assertEqual(result["best_epoch"], 1)
            self.assertTrue((root / "run" / "best_model.pt").is_file())
            log_text = (root / "run" / "training.log").read_text()
            self.assertIn("TRAINING START", log_text)
            self.assertIn("Epoch 1/1", log_text)
            self.assertIn("TRAINING COMPLETE", log_text)
            history = json.loads((root / "run" / "history.json").read_text())
            self.assertEqual(history["completed_epochs"], 1)

    def test_all_configs_load(self):
        root = Path(__file__).resolve().parents[1]
        configs = list((root / "configs/experiments").glob("*.json"))
        self.assertGreaterEqual(len(configs), 9)
        for path in configs:
            with self.subTest(path=path.name):
                self.assertEqual(load_experiment(path)["class_order"], [
                    "Communication", "Gunshot", "Footsteps", "Shelling", "Vehicle", "Helicopter", "Fighter"])


if __name__ == "__main__":
    unittest.main()
