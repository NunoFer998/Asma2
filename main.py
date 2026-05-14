import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from envs.custom_lunar_lander import CustomLunarLander


def main():
    # Usar o ambiente modificado em vez de gym.make()
    env = CustomLunarLander(render_mode="human")

    print("Espaço de observação:", env.observation_space.shape)  # deve ser (10,) ou (8,)
    print("Espaço de ações:", env.action_space)

    for episode in range(5):
        observation, info = env.reset(seed=None)  # seed=None para pad verdadeiramente aleatório
        print(f"\n--- Episódio {episode + 1} | Pad no chunk: {env._pad_idx} ---")

        for step in range(500):
            # Ação aleatória — substituir por agente treinado depois
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"  Episódio terminou ao step {step}.")
                break

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()