"""Headless (GL-free) monkeypatch for Genesis 0.3.13 on macOS without WindowServer.

Problem
-------
On this machine, shell sessions have no WindowServer access, so pyglet sees 0
screens. Genesis 0.3.13 *eagerly* builds an offscreen pyrender context whenever a
Scene is built, even with ``show_viewer=False``, because the Rasterizer is always
constructed and goes offscreen when no viewer is attached. Construction chain
(all file:line verified in .venv/lib/python3.11/site-packages/):

    WalkingEnv.__init__ -> LeggedRobotEnv -> Scene.build
      genesis/engine/scene.py:851        self._visualizer.build()
      genesis/vis/visualizer.py:179      self._rasterizer.build()
      genesis/vis/rasterizer.py:19       self._offscreen = self._viewer is None  (True: no viewer)
      genesis/vis/rasterizer.py:39       pyrender.OffscreenRenderer(pyopengl_platform=..., seg_node_map=...)
      genesis/ext/pyrender/offscreen.py:37   self._create(platform)
      genesis/ext/pyrender/offscreen.py:262  self._platform.init_context()
      genesis/ext/pyrender/platforms/pyglet_platform.py:60  pyglet.window.Window(...)  -> 0 screens
      pyglet/display/cocoa.py:32  screens[0]  -> IndexError: list index out of range

What the training path actually touches on this object
------------------------------------------------------
- Build time: nothing beyond the constructor; the instance is stored as
  ``Rasterizer._renderer`` and never used (no cameras exist).
- Step time: nothing. ``LeggedRobotEnv.apply_action`` calls
  ``scene.step(refresh_visualizer=False, update_visualizer=False)``, and
  ``Visualizer.update(force=False)`` is a no-op when ``_viewer is None``.
  ``Rasterizer.render_camera``/``update_scene`` (the only users of
  ``_renderer.make_current/render``) are reachable only through Camera objects,
  which RL training (examples/run_ppo_walking.py) never creates.
- Teardown: ``Rasterizer.destroy`` (rasterizer.py:167-168) calls
  ``self._renderer.make_current()`` and ``self._renderer.delete()`` inside a
  try/except, so the dummy must not raise there.

Patch strategy
--------------
Stub the whole ``genesis.ext.pyrender.offscreen.OffscreenRenderer`` class at the
package level. This is the highest-level GL-touching object in the no-camera
path; everything below it (pyglet platform, PyOpenGL) is then never imported or
executed, while all non-rendering code (RasterizerContext, JITRenderer with
empty primitive lists, pyrender.Scene bookkeeping) stays intact and CPU-only.
"""

from __future__ import annotations

import sys

# Make gs_env / gs_agent / gs_schemas importable: the uv editable .pth is broken
# in this environment, so prepend the source roots before any gs_env import.
_REPO_ROOT = "/Users/saichaitu/Desktop/Genesis/Genesis-Humanoid"
for _p in (
    f"{_REPO_ROOT}/src/env",
    f"{_REPO_ROOT}/src/agent",
    f"{_REPO_ROOT}/src/schemas",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_PATCHED = False
_ORIGINALS: dict[str, object] = {}


class _DummyOffscreenRenderer:
    """GL-free stand-in for genesis.ext.pyrender.offscreen.OffscreenRenderer.

    Only satisfies what the no-camera build/step/destroy path touches. Any
    attempt to actually *render* raises, so camera-based code fails loudly
    instead of silently producing garbage.
    """

    def __init__(self, point_size: float = 1.0, pyopengl_platform: str = "pyglet", seg_node_map=None):
        # Real signature: rasterizer.py:39 calls with kwargs
        # (pyopengl_platform=..., seg_node_map=...). Both are irrelevant GL-wise.
        self.point_size = float(point_size)
        self._seg_node_map = seg_node_map
        self._is_software = True  # matches "software renderer" behavior: no shadows/reflection
        self._has_valid_context = False

    # -- attributes the real class exposes (offscreen.py:41-58); some render
    #    helpers read them, keep them consistent with the real hardcoded 32.
    @property
    def viewport_width(self) -> int:
        return 32

    @property
    def viewport_height(self) -> int:
        return 32

    # -- context juggling (rasterizer.py:77/114 render path, 167-168 destroy path)
    def make_current(self) -> None:
        # No GL context exists; nothing to make current. Must not raise because
        # Rasterizer.destroy calls it inside a try/except that expects a valid
        # pairing with make_uncurrent.
        self._has_valid_context = True

    def make_uncurrent(self) -> None:
        self._has_valid_context = False

    def delete(self) -> None:
        # Rasterizer.destroy (rasterizer.py:168) calls this; no-op is fine.
        self._has_valid_context = False

    # -- rendering: the training path never calls this (no cameras). Fail loudly
    #    if someone adds a camera while headless-patched, rather than returning
    #    bogus buffers.
    def render(self, *args, **kwargs):
        raise RuntimeError(
            "g1_headless_patch: OffscreenRenderer is a GL-free dummy; rendering "
            "(cameras/sensors/RGB dumps) is not available on this headless machine."
        )

    # Keep pyrender.Renderer targets passed by add_camera alive-but-unused safe:
    # Rasterizer.remove_camera/destroy may call target.delete(); that runs on the
    # pyrender.Renderer object, not here, so nothing else is needed.


def apply_headless_patch() -> None:
    """Stub OffscreenRenderer so Scene.build never touches GL/pyglet.

    Call once BEFORE creating any env/scene. Idempotent.
    """
    global _PATCHED
    if _PATCHED:
        return

    import genesis.ext.pyrender as pyrender_pkg  # noqa: PLC0415 (needs genesis import first)
    import genesis.ext.pyrender.offscreen as offscreen_mod  # noqa: PLC0415

    # genesis/vis/rasterizer.py resolves `pyrender.OffscreenRenderer` on the
    # package namespace at call time (rasterizer.py:39), so patching the package
    # attribute is what redirects the construction. Patch the defining submodule
    # too, in case anything imports `from .offscreen import OffscreenRenderer`
    # later (e.g. fresh interpreter state or other call sites).
    _ORIGINALS["pkg"] = pyrender_pkg.OffscreenRenderer
    _ORIGINALS["mod"] = offscreen_mod.OffscreenRenderer
    pyrender_pkg.OffscreenRenderer = _DummyOffscreenRenderer
    offscreen_mod.OffscreenRenderer = _DummyOffscreenRenderer

    _PATCHED = True


def apply_registry_reward_alias() -> None:
    """Work around a PRE-EXISTING repo bug, unrelated to rendering.

    EnvArgsRegistry["g1_walk"] (src/env/gs_env/sim/envs/config/registry.py:70)
    requests reward "FeetOrientationPenalty", but the class in
    gs_env/common/rewards/g1_terms.py:218 was renamed to "FootOrientationPenalty"
    and now requires buffer name "foot_orientation", while WalkingEnv only has
    "feet_orientation". WalkingEnv cannot be constructed at all without this
    shim, with or without a GPU/GL. We register a subclass under the stale name
    that maps the env's actual buffer names onto the intended penalty. No repo
    file is modified.
    """
    import gs_env.common.rewards.g1_terms as g1_terms  # noqa: PLC0415

    if hasattr(g1_terms, "FeetOrientationPenalty"):
        return  # already fixed upstream / shimmed

    class _FeetOrientationPenalty(g1_terms.FootOrientationPenalty):
        # RewardTerm.__call__ passes state tensors positionally in
        # required_keys order, so redeclaring keys + signature is enough.
        required_keys = ("feet_orientation", "foot_contact_weighted")

        def _compute(self, feet_orientation, foot_contact_weighted):  # type: ignore
            return super()._compute(
                foot_orientation=feet_orientation,
                foot_contact_weighted=foot_contact_weighted,
            )

    g1_terms.FeetOrientationPenalty = _FeetOrientationPenalty


def main() -> None:
    import torch

    from gs_env.sim.envs.config.registry import EnvArgsRegistry
    from gs_env.sim.envs.locomotion.walking_env import WalkingEnv

    apply_headless_patch()  # MUST run before env creation (Scene.build)
    apply_registry_reward_alias()  # pre-existing registry/class-name mismatch

    env = WalkingEnv(
        args=EnvArgsRegistry["g1_walk"],
        num_envs=2,
        show_viewer=False,
        device=torch.device("cpu"),
    )

    actor_obs, _critic_obs = env.get_observations()
    print(f"obs shape: {tuple(actor_obs.shape)}")

    action_dim = env.action_space.shape[-1]
    assert action_dim == 29, f"expected action dim 29, got {action_dim}"

    rewards = []
    for i in range(5):
        action = torch.zeros(2, action_dim, device=torch.device("cpu"))
        obs, reward, terminated, truncated, _extra = env.step(action)
        assert not torch.isnan(obs).any(), f"NaN in obs at step {i}"
        assert not torch.isnan(reward).any(), f"NaN in reward at step {i}"
        rewards.append(reward.mean().item())

    print(f"reward mean over 5 steps: {sum(rewards) / len(rewards):.6f}")

    # Extra robustness: 30 more steps, no NaN in obs
    for i in range(30):
        obs, reward, _term, _trunc, _extra = env.step(torch.zeros(2, action_dim))
        assert not torch.isnan(obs).any(), f"NaN in obs at extra step {i}"

    print("G1 HEADLESS PASS")


if __name__ == "__main__":
    main()
