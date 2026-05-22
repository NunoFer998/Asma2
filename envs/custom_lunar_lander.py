import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE

try:
    from Box2D.b2 import edgeShape
except ImportError as e:
    raise ImportError("Box2D não está instalado. Corre: pip install swig && pip install 'gymnasium[box2d]'") from e


class CustomLunarLander(LunarLander):
    """
    LunarLander modificado com landing pad em posição aleatória a cada episódio.
    - O pad pode aparecer em qualquer um dos chunks 2 a 8 (evita as bordas)
    - As flags amarelas movem-se visualmente
    - O vetor de estado é expandido de 8 para 10 dimensões:
        [0] x do lander relativo ao pad (não ao centro do ecrã)
        [1] y do lander
        [2] vel_x
        [3] vel_y
        [4] ângulo
        [5] vel_angular
        [6] contacto perna esquerda
        [7] contacto perna direita
        [8] posição x do pad (normalizada, nova!)
        [9] posição y do pad (normalizada, nova!)
    """

    # Gymnasium defines LEG_DOWN = 18 in lunar_lander.py.
    # We define it here explicitly so the env won't silently break
    # if that module-level constant is ever renamed or changed.
    _LEG_DOWN: int = 18

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )
        self._custom_prev_shaping = None  # tracks our corrective shaping potential

    def reset(self, *, seed=None, options=None):
        # Inicializar helipad_center_x para evitar AttributeError durante super().reset()
        # NOTA (Bug 3): durante super().reset() o parent calcula o shaping inicial
        # usando o centro do ecrã como pad. Isto é inofensivo porque o parent
        # reinicia prev_shaping = None no início de cada episódio.
        self.helipad_center_x = (VIEWPORT_W / SCALE) / 2  # Default: centro
        self.helipad_y = (VIEWPORT_H / SCALE) / 4  # Default height
        
        # Chama o reset original — cria o mundo, lander, terreno com pad central
        obs, info = super().reset(seed=seed, options=options)

        # Reconstrói o terreno com pad aleatório (substitui o que o super() criou)
        self._rebuild_terrain_random()

        # Reset do corrective shaping tracker (Bug 1 fix)
        self._custom_prev_shaping = None

        # Recalcula a observação com o novo estado (pad na posição correta)
        obs = self._get_custom_obs()
        return obs, info

    def _rebuild_terrain_random(self):
        """Destrói o terreno original e cria um novo com pad em chunk aleatório."""
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        CHUNKS = 11

        # Destruir terreno criado pelo super()
        self.world.DestroyBody(self.moon)
        self.sky_polys = []

        # Gerar alturas aleatórias para o terreno
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        # ✅ Chunk aleatório para o pad (entre 2 e 8, evita bordas)
        pad_idx = int(self.np_random.integers(2, 9))
        self._pad_idx = pad_idx

        # Definir posição do helipad
        self.helipad_x1 = chunk_x[pad_idx - 1]
        self.helipad_x2 = chunk_x[pad_idx + 1]
        self.helipad_y  = H / 4
        self.helipad_center_x = chunk_x[pad_idx]  # centro do pad em coordenadas do mundo

        # Aplanar 5 chunks à volta do pad escolhido
        for offset in [-2, -1, 0, 1, 2]:
            height[pad_idx + offset] = self.helipad_y

        # Suavizar terreno
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        # Recriar corpo físico da lua
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
        """Constrói o vetor de estado com 10 dimensões (8 originais + posição do pad)."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        state = [
            # [0] x relativo ao pad (não ao centro do ecrã — esta é a correção chave)
            (pos.x - self.helipad_center_x) / (VIEWPORT_W / SCALE / 2),
            # [1] y relativo ao pad (Bug 4: usa _LEG_DOWN em vez de magic number)
            (pos.y - (self.helipad_y + self._LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            # [2,3] velocidades
            vel.x * (VIEWPORT_W / SCALE / 2) / 50,
            vel.y * (VIEWPORT_H / SCALE / 2) / 50,
            # [4,5] ângulo e velocidade angular
            self.lander.angle,
            20.0 * self.lander.angularVelocity / 50,
            # [6,7] contacto das pernas
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            # [8] posição x do pad normalizada (-1 a 1)
            (self.helipad_center_x - W / 2) / (W / 2),
            # [9] posição y do pad normalizada
            self.helipad_y / (H / 2) - 1.0,
        ]
        return np.array(state, dtype=np.float32)

    # ── Precision landing bonus parameters ──────────────────────
    #   bonus(d) = PRECISION_MAX_BONUS * exp(-d² / (2 * σ²))
    #   where d = horizontal distance from pad centre (world units)
    #
    #   With σ = 0.4 (≈ half a chunk width):
    #     d = 0.0  → +40.0   (bullseye)
    #     d = 0.4  → +24.3   (edge of pad)
    #     d = 1.0  → + 1.2   (one chunk away — nearly zero)
    PRECISION_MAX_BONUS: float = 40.0
    PRECISION_SIGMA: float = 0.4

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Substituir os primeiros 8 valores da obs pelo estado corrigido
        # (o super().step() ainda calcula x relativo ao centro — corrigimos aqui)
        custom_obs = self._get_custom_obs()

        # ════════════════════════════════════════════════════════════
        #  Bug 1 fix: Corrective shaping  (potential-based)
        # ════════════════════════════════════════════════════════════
        # The parent's dense shaping penalises distance from the SCREEN
        # CENTER (x = W/2).  We need the gradient to point at the actual
        # PAD CENTER (self.helipad_center_x).
        #
        # Corrective potential:
        #   Φ(s) = -100 * (|x_pad_norm| - |x_screen_norm|)
        #
        # Applied as:  reward += Φ(s) - Φ(s_prev)     (telescoping)
        #
        # Net effect on the agent's signal:
        #   parent gave  ∝ -|x_screen|   per step
        #   correction   ∝ -(|x_pad| - |x_screen|)
        #   total        ∝ -|x_pad|      ← what we want
        # ────────────────────────────────────────────────────────────
        pos_x = self.lander.position.x
        W = VIEWPORT_W / SCALE

        x_screen_norm = (pos_x - W / 2) / (W / 2)
        x_pad_norm    = (pos_x - self.helipad_center_x) / (W / 2)

        corrective_shaping = -100.0 * (abs(x_pad_norm) - abs(x_screen_norm))

        if self._custom_prev_shaping is not None:
            reward += corrective_shaping - self._custom_prev_shaping
        self._custom_prev_shaping = corrective_shaping

        # ════════════════════════════════════════════════════════════
        #  Bug 2 fix + Precision landing bonus
        # ════════════════════════════════════════════════════════════
        # Crash detection now uses self.game_over (set by the parent's
        # ContactDetector when the lander HULL touches the ground)
        # instead of a fragile reward threshold.
        # ────────────────────────────────────────────────────────────
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

        # ── Debug info (verify the shaping fix is working) ───────
        info["x_pad_dist"] = round(float(abs(pos_x - self.helipad_center_x)), 4)

        return custom_obs, reward, terminated, truncated, info