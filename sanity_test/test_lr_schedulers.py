import math
import unittest

from paraphrase_generation.lr_schedulers import (
    ConstantLearningRate,
    CosineDecay,
    InverseSquareRoot,
    LinearDecay,
    MetricDependent,
    StepDecay,
)


class ConstantLearningRateTest(unittest.TestCase):
    def test_learning_rate_does_not_change(self):
        scheduler = ConstantLearningRate(2e-5)

        for _ in range(5):
            self.assertEqual(scheduler.lr(), 2e-5)
            scheduler.step()

    def test_rejects_invalid_learning_rates(self):
        for value in (-1.0, math.inf, math.nan):
            with self.subTest(initial_lr=value):
                with self.assertRaises(ValueError):
                    ConstantLearningRate(value)

        for value in (-1.0, math.inf, math.nan):
            with self.subTest(min_lr=value):
                with self.assertRaises(ValueError):
                    ConstantLearningRate(1.0, min_lr=value)

        with self.assertRaises(ValueError):
            ConstantLearningRate(1.0, min_lr=1.1)


class StepDecayTest(unittest.TestCase):
    def test_decays_at_step_boundaries_and_respects_floor(self):
        scheduler = StepDecay(1.0, step_size=2, gamma=0.5, min_lr=0.2)
        expected = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.2]

        for expected_lr in expected:
            self.assertAlmostEqual(scheduler.lr(), expected_lr)
            scheduler.step()

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            StepDecay(1.0, step_size=0)
        with self.assertRaises(ValueError):
            StepDecay(1.0, step_size=1, gamma=0.0)
        with self.assertRaises(ValueError):
            StepDecay(1.0, step_size=1, gamma=1.1)


class CosineDecayTest(unittest.TestCase):
    def test_reaches_minimum_after_total_steps(self):
        scheduler = CosineDecay(1.0, total_steps=4, min_lr=0.2)
        expected = [1.0, 0.8828427, 0.6, 0.3171573, 0.2]

        for expected_lr in expected:
            self.assertAlmostEqual(scheduler.lr(), expected_lr, places=6)
            scheduler.step()

        scheduler.step()
        self.assertEqual(scheduler.lr(), 0.2)

    def test_rejects_non_positive_total_steps(self):
        with self.assertRaises(ValueError):
            CosineDecay(1.0, total_steps=0)
        with self.assertRaises(ValueError):
            CosineDecay(1.0, total_steps=1, min_lr=-0.1)

    def test_rejects_metric_argument(self):
        scheduler = CosineDecay(1.0, total_steps=2)
        with self.assertRaises(ValueError):
            scheduler.step(metric=0.5)


class LinearDecayTest(unittest.TestCase):
    def test_decays_linearly_and_stays_at_floor(self):
        scheduler = LinearDecay(1.0, total_steps=4, min_lr=0.2)
        expected = [1.0, 0.8, 0.6, 0.4, 0.2]

        for expected_lr in expected:
            self.assertAlmostEqual(scheduler.lr(), expected_lr)
            scheduler.step()

        scheduler.step()
        self.assertEqual(scheduler.lr(), 0.2)

    def test_rejects_non_positive_total_steps(self):
        with self.assertRaises(ValueError):
            LinearDecay(1.0, total_steps=0)


class InverseSquareRootTest(unittest.TestCase):
    def test_decays_as_inverse_square_root_without_warmup(self):
        scheduler = InverseSquareRoot(1.0)

        self.assertAlmostEqual(scheduler.lr(), 1.0)
        scheduler.step()
        self.assertAlmostEqual(scheduler.lr(), 1.0 / math.sqrt(2.0))
        scheduler.step()
        self.assertAlmostEqual(scheduler.lr(), 1.0 / math.sqrt(3.0))

    def test_warmup_reaches_initial_rate_then_decays(self):
        scheduler = InverseSquareRoot(1.0, warmup_steps=2)
        expected = [0.5, 1.0, math.sqrt(2.0 / 3.0)]

        for expected_lr in expected:
            self.assertAlmostEqual(scheduler.lr(), expected_lr)
            scheduler.step()

    def test_rejects_negative_warmup(self):
        with self.assertRaises(ValueError):
            InverseSquareRoot(1.0, warmup_steps=-1)


class MetricDependentTest(unittest.TestCase):
    def test_reduces_after_patience_is_exceeded(self):
        scheduler = MetricDependent(1.0, factor=0.5, patience=1, min_lr=0.2)

        scheduler.step(0.5)  # Establishes the best metric.
        scheduler.step(0.4)  # First bad epoch: patience not exceeded.
        self.assertEqual(scheduler.lr(), 1.0)
        scheduler.step(0.4)  # Second bad epoch: reduce the learning rate.
        self.assertEqual(scheduler.lr(), 0.5)

        scheduler.step(0.3)
        scheduler.step(0.3)
        self.assertEqual(scheduler.lr(), 0.25)
        scheduler.step(0.2)
        scheduler.step(0.2)
        self.assertEqual(scheduler.lr(), 0.2)

    def test_improvement_resets_bad_epoch_count(self):
        scheduler = MetricDependent(1.0, factor=0.5, patience=1)

        scheduler.step(0.5)
        scheduler.step(0.4)
        scheduler.step(0.6)  # Improvement resets patience.
        scheduler.step(0.5)
        self.assertEqual(scheduler.lr(), 1.0)

    def test_rejects_non_finite_metrics(self):
        scheduler = MetricDependent(1.0)
        for metric in (math.inf, -math.inf, math.nan):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError):
                    scheduler.step(metric)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            MetricDependent(1.0, factor=0.0)
        with self.assertRaises(ValueError):
            MetricDependent(1.0, factor=1.0)
        with self.assertRaises(ValueError):
            MetricDependent(1.0, patience=-1)
        with self.assertRaises(ValueError):
            MetricDependent(1.0, threshold=-1.0)


if __name__ == "__main__":
    unittest.main()
