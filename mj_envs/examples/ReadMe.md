# Examples
Here a set of examples on how to use different MysSimSuite models and non-stationarities.

* [TestMuscle Fatigue](#Test-Muscle-Fatigue)
* [Test Sarcopenia](#Test-Sarcopenia)
* [Test Virtual tendon transfer](#Test-Virtual-tendon-transfer)
* [Test Physical tendon transfer](#Test-Physical-tendon-transfer)
* [Test Exoskeleton](#Test-Exoskeleton)
* [Resume Learning of policies](#Resume-Learning-of-policies)
*

## Test Muscle Fatigue
This example shows how to add fatigue to a model. It tests random actions on a model without and then with muscle fatigue.
```python
import myoSuite
import gym
env = gym.make('ElbowPose1D6MRandom-v0')
env.reset()
env.sim.render(mode='window')
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
# Add muscle fatigue
env.env.muscle_condition = 'fatigue'
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.close()
```

## Test Sarcopenia
This example shows how to add sarcopenia or muscle weakness to a model. It tests random actions on a model without and then with muscle weakness.
```python
import mj_envs
import gym
env = gym.make('ElbowPose1D6MRandom-v0')
env.reset()
  env.sim.render(mode='window')
for _ in range(1000):
  env.step(env.action_space.sample()) # take a random action
# Add muscle weakness
env.env.muscle_condition = 'weakness'
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.close()
```

## Test Virtual tendon transfer
This example shows how to test a model with virtual tendon transfer. It tests random adctions on a model without and then with indext to thumb tendon transfer .
```python
import mj_envs
import gym
env = gym.make('XX')
env.reset()
  env.sim.render(mode='window')
for _ in range(1000):
  env.step(env.action_space.sample()) # take a random action
# Add tendon transfer
env.env.muscle_condition = '???'
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.close()
```

## Test Physical tendon transfer
This example shows how load a model with physical tendon transfer.

```python
import myoSuite
import gym
env = gym.make('XXX')
env.reset()
env.sim.render(mode='window')
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.close()
```

## Test Exoskeleton
This example shows how load a model with an exoskeleton and switch it off.

```python
import myoSuite
import gym
env = gym.make('ElbowPose1D6MExoRandom-v0')
env.reset()
env.sim.render(mode='window')
for _ in range(1000):
  env.step(env.action_space.sample()) # take a random action
# switch exoskeleton off
for _ in range(1000):
    act = env.action_space.sample()
    act[0] = 0
    env.step(act) # take a random without exoskeleton
env.close()
```

## Resume Learning of policies
TODO
```bash
python ../
```
