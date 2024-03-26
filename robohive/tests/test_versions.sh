pip uninstall -y gym
pip uninstall -y gymnasium
pip uninstall -y stable-baselines3

echo "=================== Testing gym==0.13 ==================="
pip install gym==0.13
python tests/test_all.py
pip uninstall -y gym

echo "=================== Testing gym==0.26.2 ==================="
pip install gym==0.26.2
python tests/test_all.py
pip uninstall -y gym

echo "=================== Testing gymnasium ==================="
pip install gymnasium
python tests/test_all.py

echo "=================== Testing Stable Baselines ==================="
pip install stable-baselines3
python tests/test_sb.py
pip uninstall -y gymnasium
pip uninstall -y stable-baselines3
