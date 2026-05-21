from __future__ import annotations

import numpy as np
from gymnasium import Wrapper

try:
    import pygame
except ImportError as e:
    raise ImportError("pygame is required to render the fuel bar. Install gymnasium[box2d].") from e


class FiniteFuelWrapper(Wrapper):
    """Adds a strict finite-fuel system on top of a LunarLander environment.

    The wrapper keeps the base environment rewards and dynamics intact, but it
    cuts engine thrust once the fuel tank is empty and renders a fuel bar in
    human mode.
    """

    def __init__(
        self,
        env,
        max_fuel: float = 1000.0,
        main_engine_cost: float = 30.0,
        side_engine_cost: float = 10.0,
    ):
        super().__init__(env)
        self.max_fuel = float(max_fuel)
        self.main_engine_cost = float(main_engine_cost)
        self.side_engine_cost = float(side_engine_cost)
        self.current_fuel = self.max_fuel

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.current_fuel = self.max_fuel
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
        return observation, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()

        if self.render_mode == "rgb_array":
            if frame is None:
                return None
            return self._overlay_bar_on_frame(frame)

        self._draw_fuel_bar()
        pygame.display.flip()

        return frame

    def _fuel_cost_for_action(self, action: int) -> float:
        if action == 2:
            return self.main_engine_cost
        if action in (1, 3):
            return self.side_engine_cost
        return 0.0

    def _get_surface(self):
        base_env = self.env.unwrapped
        if self.render_mode == "human":
            return getattr(base_env, "screen", None) or getattr(base_env, "window", None)
        return getattr(base_env, "surf", None) or getattr(base_env, "screen", None)

    def _draw_fuel_bar(self) -> None:
        surface = self._get_surface()
        if surface is None:
            return

        fuel_ratio = 0.0 if self.max_fuel <= 0 else max(0.0, min(1.0, self.current_fuel / self.max_fuel))
        bar_width = 160
        bar_height = 18
        x = 12
        y = 12
        border_color = (245, 245, 245)
        background_color = (20, 20, 20)
        bar_color = (
            int(255 * (1.0 - fuel_ratio)),
            int(255 * fuel_ratio),
            0,
        )

        pygame.draw.rect(surface, background_color, (x, y, bar_width, bar_height))
        pygame.draw.rect(surface, border_color, (x, y, bar_width, bar_height), width=2)

        inner_width = max(0, int((bar_width - 4) * fuel_ratio))
        if inner_width > 0:
            pygame.draw.rect(surface, bar_color, (x + 2, y + 2, inner_width, bar_height - 4))

    def _overlay_bar_on_frame(self, frame):
        bar_width = 160
        bar_height = 18
        x = 12
        y = 12
        border_color = np.array([245, 245, 245], dtype=np.uint8)
        background_color = np.array([20, 20, 20], dtype=np.uint8)

        output = np.array(frame, copy=True)
        height, width = output.shape[:2]
        x2 = min(width, x + bar_width)
        y2 = min(height, y + bar_height)
        if x >= x2 or y >= y2:
            return output

        output[y:y2, x:x2] = background_color

        border_thickness = 2
        output[y:y + border_thickness, x:x2] = border_color
        output[max(y2 - border_thickness, y):y2, x:x2] = border_color
        output[y:y2, x:x + border_thickness] = border_color
        output[y:y2, max(x2 - border_thickness, x):x2] = border_color

        fuel_ratio = 0.0 if self.max_fuel <= 0 else max(0.0, min(1.0, self.current_fuel / self.max_fuel))
        inner_width = max(0, int((bar_width - 4) * fuel_ratio))
        if inner_width > 0:
            bar_color = np.array([
                int(255 * (1.0 - fuel_ratio)),
                int(255 * fuel_ratio),
                0,
            ], dtype=np.uint8)
            fill_x2 = min(width, x + 2 + inner_width)
            fill_y2 = min(height, y + bar_height - 2)
            output[y + 2:fill_y2, x + 2:fill_x2] = bar_color

        return output