from __future__ import annotations

import copy
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper

import pygame


class FiniteFuelWrapper(Wrapper):
    """Finite-fuel system with HUD overlay."""

    def __init__(
        self,
        env,
        max_fuel: float = 4000.0,
        main_engine_cost: float = 12.0,
        side_engine_cost: float = 7.0,
    ):
        assert env.render_mode in (None, "rgb_array"), (
            f"FiniteFuelWrapper requires render_mode=None or 'rgb_array'. "
            f"Got: {env.render_mode!r}"
        )
        super().__init__(env)
        self.max_fuel = float(max_fuel)
        self.main_engine_cost = float(main_engine_cost)
        self.side_engine_cost = float(side_engine_cost)
        self.current_fuel = self.max_fuel

        self._window = None
        self._clock = None
        self._screen_size = None

        self.metadata = copy.deepcopy(env.metadata)
        if "human" not in self.metadata.get("render_modes", []):
            self.metadata.setdefault("render_modes", []).append("human")

    @property
    def render_mode(self):
        return "human"

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.current_fuel = self.max_fuel
        if self.env.render_mode == "rgb_array":
            self._render_frame()
        return observation, info

    def step(self, action):
        action_int = int(np.asarray(action).item())

        if self.current_fuel <= 0:
            executed_action = 0
        else:
            executed_action = action_int
            fuel_cost = self._fuel_cost_for_action(executed_action)
            if fuel_cost > 0:
                self.current_fuel = max(0.0, self.current_fuel - fuel_cost)

        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        info = dict(info)
        info["current_fuel"] = self.current_fuel
        info["max_fuel"] = self.max_fuel
        info["executed_action"] = executed_action

        if self.env.render_mode == "rgb_array":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def _render_frame(self):
        """Render the frame with fuel HUD overlay."""
        if self.env.render_mode != "rgb_array":
            return
        frame = self.env.render()  # numpy array (H, W, 3)
        if frame is None:
            return

        frame = self._overlay_bar_on_frame(frame)
        rgb_array = np.transpose(frame, axes=(1, 0, 2))

        if self._screen_size is None:
            self._screen_size = rgb_array.shape[:2]

        if self._window is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Lunar Lander — Finite Fuel")
            self._window = pygame.display.set_mode(self._screen_size)

        if self._clock is None:
            self._clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        self._window.blit(surf, (0, 0))
        pygame.event.pump()
        self._clock.tick(self.metadata.get("render_fps", 50))
        pygame.display.flip()

    def close(self):
        super().close()
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fuel_cost_for_action(self, action: int) -> float:
        if action == 2:
            return self.main_engine_cost
        if action in (1, 3):
            return self.side_engine_cost
        return 0.0

    def _overlay_bar_on_frame(self, frame: np.ndarray) -> np.ndarray:
        bar_width, bar_height = 160, 18
        x, y = 12, 12
        border_color = np.array([245, 245, 245], dtype=np.uint8)
        background_color = np.array([20, 20, 20], dtype=np.uint8)

        output = np.array(frame, copy=True)
        height, width = output.shape[:2]
        x2 = min(width, x + bar_width)
        y2 = min(height, y + bar_height)

        output[y:y2, x:x2] = background_color

        t = 2  # border thickness
        output[y:y+t, x:x2] = border_color
        output[max(y2-t, y):y2, x:x2] = border_color
        output[y:y2, x:x+t] = border_color
        output[y:y2, max(x2-t, x):x2] = border_color

        fuel_ratio = max(0.0, min(1.0, self.current_fuel / self.max_fuel)) if self.max_fuel > 0 else 0.0
        inner_width = max(0, int((bar_width - 4) * fuel_ratio))
        if inner_width > 0:
            bar_color = np.array([
                int(255 * (1.0 - fuel_ratio)),
                int(255 * fuel_ratio),
                0,
            ], dtype=np.uint8)
            output[y+2:min(height, y2-2), x+2:min(width, x+2+inner_width)] = bar_color

        return output