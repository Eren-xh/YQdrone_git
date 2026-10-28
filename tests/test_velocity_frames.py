"""Check velocity axes against MuJoCo kinematics and its local velocimeter."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from controllers.approach_controller import ApproachController
from envs.xh1_aviary import XH1Aviary


class VelocityFrameTests(unittest.TestCase):
    def setUp(self):
        self.env = XH1Aviary()
        self.addCleanup(self.env.close)

    def set_motion(self, rpy, linear, angular=(0.0, 0.0, 0.0)):
        self.env.reset(options={"position": [2.0, 0.0, 1.0], "rpy": rpy})
        self.env.data.qvel[:3] = linear
        self.env.data.qvel[3:6] = angular
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_known_world_translation_at_different_attitudes(self):
        velocity = np.array([0.7, -0.4, 0.2])
        for attitude in ([0, 0, 0], [0, 0, np.pi / 2], [.4, -.6, 1.1]):
            with self.subTest(attitude=attitude):
                self.set_motion(attitude, velocity)
                np.testing.assert_allclose(self.env.velocity_world, velocity, atol=1e-12)
                np.testing.assert_allclose(
                    self.env.velocity_body, self.env._sensor("flow_velocity_proxy"), atol=1e-12)
        # Default FRD axes are world -X, +Y, -Z: a world +X motion
        # therefore has a negative forward component at this installation.
        self.set_motion([0, 0, 0], velocity)
        np.testing.assert_allclose(self.env.velocity_body, [-.7, -.4, -.2], atol=1e-12)

    def test_imu_point_velocity_includes_rotation_about_offset_root(self):
        self.set_motion([.5, -.4, .9], [.7, -.4, .2], [.8, -.6, .5])
        jacobian = np.zeros((3, self.env.model.nv))
        mujoco.mj_jacSite(self.env.model, self.env.data, jacobian, None, self.env.imu_site_id)
        expected_world = jacobian @ self.env.data.qvel
        self.assertGreater(np.linalg.norm(expected_world - self.env.data.qvel[:3]), .001)
        np.testing.assert_allclose(self.env.velocity_world, expected_world, atol=1e-12)
        np.testing.assert_allclose(
            self.env.velocity_body, self.env._sensor("flow_velocity_proxy"), atol=1e-12)

    def test_aerodynamics_receives_correct_axes_and_drag_opposes_motion(self):
        self.set_motion([.3, -.5, .8], [.7, -.4, .2], [.4, -.3, .2])
        original_apply = self.env.aerodynamics.apply
        calls = []

        def inspect_apply(*args, **kwargs):
            velocity_body, velocity_world = args[6:8]
            np.testing.assert_allclose(velocity_world, self.env._sensor("imu_vel"), atol=1e-12)
            np.testing.assert_allclose(
                velocity_body, self.env._sensor("flow_velocity_proxy"), atol=1e-12)
            drag_body = self.env.aerodynamics.blade_drag_body(args[5], velocity_body)
            drag_world = self.env._body_to_world() @ drag_body
            self.assertLess(float(np.dot(drag_world, velocity_world)), 0.0)
            calls.append(1)
            return original_apply(*args, **kwargs)

        with patch.object(self.env.aerodynamics, "apply", side_effect=inspect_apply):
            _, info = self.env._simulate_command(.179 * 9.81, np.zeros(3))
        self.assertEqual(len(calls), self.env.sim_steps_per_ctrl)
        np.testing.assert_allclose(info["velocity_world"], self.env._sensor("imu_vel"), atol=1e-12)
        np.testing.assert_allclose(
            info["velocity_body"], self.env._sensor("flow_velocity_proxy"), atol=1e-12)
        # Logged end-state snapshots must survive later sensor updates.
        saved = info["velocity_world"].copy()
        self.env.data.qvel[:] = 0.
        mujoco.mj_forward(self.env.model, self.env.data)
        np.testing.assert_array_equal(info["velocity_world"], saved)

    def test_world_velocity_damping_does_not_depend_on_body_heading(self):
        accelerations = []
        for attitude in ([0, 0, 0], [0, 0, np.pi / 2], [.4, -.6, 1.1]):
            self.set_motion(attitude, [.1, -.2, .3])
            controller = ApproachController()
            controller.set_target(self.env.position.copy())
            _, _, info = controller.command(self.env)
            accelerations.append(info["approach_desired_acceleration"])
        for acceleration in accelerations[1:]:
            np.testing.assert_allclose(acceleration, accelerations[0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
