<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Project Title</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Gear placement training
    <br> 
</p>

## 📝 Table of Contents

- [Description](#description)
- [Getting Started](#getting_started)
- [Test](#tests)
- [Authors](#authors)

## 🧐 Description <a name = "description"></a>

Torque based training:
- simulated torque value as input of agent
- policy: DDPG, PPO

Vision based training:
- precaptured image as input
- policy:DQN

## 🏁 Getting Started <a name = "getting_started"></a>

### Prerequisites
```
stable_baseline3
pyglet
gym
opencv
```

### Installing

```
git clone https://github.com/Adiming/rl_image_train.git
```

## 🔧 Running the tests <a name = "tests"></a>

### Vision based 

#### Training
make sure the env create line(210) and img train line (211) in file train_img_dqn,py is uncommand
```
python3 train_img_dqn.py
```
#### Output trajectory
make sure the env create line(210) and img train line (211) in file train_img_dqn,py is command and uncommand predict_and_write() line(212)
```
python3 train_img_dqn.py
```

### Torque based 

#### Training
make sure test_train() in file train_torque is uncommand
```
python3 train_torque.py
```
#### Output trajectory
make sure the test_train() in file train_torque is command and uncommand predict_and_write() line(212)
```
python3 train_torque.py
```
### Plot trajectory
make sure given the correct csv file name
```
python3 plot_trajctory.py
```

## ✍️ Authors <a name = "authors"></a>

- [@Junjie MIng](https://github.com/kylelobo) - Idea & Initial work


