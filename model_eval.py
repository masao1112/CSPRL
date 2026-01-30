from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import pickle
import env_plus as ev
import seaborn as sns
import matplotlib.pyplot as plt

"""
Generate a charging plan based on the model.
"""
# Instantiate the env
location = "DongDa"

graph_file = "Graph/" + location + "/" + location + ".graphml"
node_file = "Graph/" + location + "/nodes_extended_" + location + ".txt"
plan_file = "Graph/" + location + "/existingplan_" + location + ".pkl"

env = ev.StationPlacement(graph_file, node_file, plan_file)
log_dir = f"tmp_{location}/"

"""
Ab hier kommt die Evaluation.
"""
print("Evaluation for best model")
env = Monitor(env, log_dir)  # new environment for evaluation

model = DQN.load(log_dir + "best_model_" + location + "_29200.zip")

obs, _ = env.reset()
done = False
best_plan, best_node_list = None, None
action_history = []
while not done:
    action, _states = model.predict(obs, deterministic=True)
    action_history.append(action.item())
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if done:
        best_node_list, best_plan = env.render()

sns.countplot(x=action_history)
plt.title('Frequency of Chosen Actions')
plt.show()

pickle.dump(best_plan, open("Results/" + location + "/plan_RL.pkl", "wb"))
with open("Results/" + location + "/nodes_RL.txt", 'w') as file:
    file.write(str(best_node_list))
