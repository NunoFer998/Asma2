import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE

try:
    from Box2D.b2 import edgeShape
except ImportError as e:
    raise ImportError("Box2D não está instalado. Corre: pip install swig && pip install 'gymnasium[box2d]'") from e


class CustomLunarLander(LunarLander):
    """LunarLander with a randomised landing pad and 10-dim observation space."""

    _LEG_DOWN: int = 18  # local copy of Gymnasium's LEG_DOWN constant

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )
        self._custom_prev_shaping = None

    def reset(self, *, seed=None, options=None):
        self.helipad_center_x = (VIEWPORT_W / SCALE) / 2
        self.helipad_y = (VIEWPORT_H / SCALE) / 4

        obs, info = super().reset(seed=seed, options=options)
        self._rebuild_terrain_random()
        self._custom_prev_shaping = None
        obs = self._get_custom_obs()
        return obs, info

    def _rebuild_terrain_random(self):
        """Rebuild terrain with the landing pad at a random chunk."""
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        CHUNKS = 11

        self.world.DestroyBody(self.moon)
        self.sky_polys = []

        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        pad_idx = int(self.np_random.integers(2, 9))
        self._pad_idx = pad_idx

        self.helipad_x1 = chunk_x[pad_idx - 1]
        self.helipad_x2 = chunk_x[pad_idx + 1]
        self.helipad_y  = H / 4
        self.helipad_center_x = chunk_x[pad_idx] 

        for offset in [-2, -1, 0, 1, 2]:
            height[pad_idx + offset] = self.helipad_y

        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

    def _get_custom_obs(self):
        """Build the 10-dim observation vector."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        state = [
            (pos.x - self.helipad_center_x) / (W / 2),
            (pos.y - (self.helipad_y + self._LEG_DOWN / SCALE)) / (H / 2),
            vel.x * (W / 2) / 50,
            vel.y * (H / 2) / 50,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / 50,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (self.helipad_center_x - W / 2) / (W / 2),
            self.helipad_y / (H / 2) - 1.0,
        ]
        return np.array(state, dtype=np.float32)

    PRECISION_MAX_BONUS: float = 40.0
    PRECISION_SIGMA: float = 0.4

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        custom_obs = self._get_custom_obs()

        # Corrective shaping: redirect the parent's dense signal from
        # screen centre to the actual pad centre (potential-based).
        pos_x = self.lander.position.x
        W = VIEWPORT_W / SCALE

        x_screen_norm = (pos_x - W / 2) / (W / 2)
        x_pad_norm    = (pos_x - self.helipad_center_x) / (W / 2)

        corrective_shaping = -100.0 * (abs(x_pad_norm) - abs(x_screen_norm))

        if self._custom_prev_shaping is not None:
            reward += corrective_shaping - self._custom_prev_shaping
        self._custom_prev_shaping = corrective_shaping

        # Precision landing bonus (Gaussian, uses game_over for crash detection).
        if terminated and not truncated:
            landed_safely = (
                not self.game_over
                and self.legs[0].ground_contact
                and self.legs[1].ground_contact
            )

            if landed_safely:
                dx = self.lander.position.x - self.helipad_center_x
                precision_bonus = self.PRECISION_MAX_BONUS * np.exp(
                    -(dx ** 2) / (2 * self.PRECISION_SIGMA ** 2)
                )
                reward += precision_bonus
                info["precision_bonus"] = round(float(precision_bonus), 4)
                info["landing_offset"]  = round(float(dx), 4)

        info["x_pad_dist"] = round(float(abs(pos_x - self.helipad_center_x)), 4)

        return custom_obs, reward, terminated, truncated, info