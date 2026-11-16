import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.safety import Blocked, ConstraintLayer


class DummyProximity:
    def partner_of(self, robot):
        return []


class ConstraintLayerAuthorizationTests(unittest.TestCase):
    def test_execute_accepts_granted_token(self):
        layer = ConstraintLayer(DummyProximity())
        token = layer.request("go2", "drive")
        executed = []
        self.assertEqual(layer.stats()["executed"], 0)

        layer.execute(token, executed)

        self.assertEqual(executed, [token])
        self.assertEqual(layer.stats()["executed"], 1)

    def test_execute_rejects_blocked_token(self):
        layer = ConstraintLayer(DummyProximity())
        token = layer.request("go2", "drive", conflict=["g1"])
        executed = []

        with self.assertRaises(Blocked):
            layer.execute(token, executed)

        self.assertEqual(executed, [])
        self.assertEqual(layer.stats()["executed"], 0)

    def test_execute_rejects_unknown_token(self):
        layer = ConstraintLayer(DummyProximity())
        executed = []

        with self.assertRaises(Blocked):
            layer.execute(999, executed)

        self.assertEqual(executed, [])
        self.assertEqual(layer.stats()["executed"], 0)
