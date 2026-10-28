"""Small XML/Gymnasium smoke test for the XH1 project."""

from envs.xh1_aviary import XH1Aviary


def test_model() -> None:
    env = XH1Aviary()
    observation, info = env.reset(seed=0)
    assert observation.shape == (13,)
    assert abs(float(env.model.body_mass.sum()) - 0.179) < 1e-9
    print("XH1 model ready", observation.shape, info)
    env.close()


if __name__ == "__main__":
    test_model()
