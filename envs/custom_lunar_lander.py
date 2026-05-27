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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        # Inicializar helipad_center_x para evitar AttributeError durante super().reset()
        self.helipad_center_x = (VIEWPORT_W / SCALE) / 2  # Default: centro
        self.helipad_y = (VIEWPORT_H / SCALE) / 4  # Default height
        
        # Chama o reset original — cria o mundo, lander, terreno com pad central
        obs, info = super().reset(seed=seed, options=options)

        # Reconstrói o terreno com pad aleatório (substitui o que o super() criou)
        self._rebuild_terrain_random()

        # Recalcula a observação com o novo estado (pad na posição correta)
        obs = self._get_custom_obs()
        return obs, info

    def _rebuild_terrain_random(self):
        """Destrói o terreno original e cria um novo com pad em chunk aleatório."""
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
        """Constrói o vetor de estado com 10 dimensões (8 originais + posição do pad)."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        state = [
            (pos.x - self.helipad_center_x) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + 18 / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / 50,
            vel.y * (VIEWPORT_H / SCALE / 2) / 50,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / 50,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (self.helipad_center_x - W / 2) / (W / 2),
            self.helipad_y / (H / 2) - 1.0,
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

      
        custom_obs = self._get_custom_obs()

        return custom_obs, reward, terminated, truncated, info