pip uninstall -y gym
pip uninstall -y gymnasium

echo "=================== Testing gym==0.13 ==================="
pip install gym==0.13
python tests/test_arms.py
python tests/test_examine_env.py
python tests/test_examine_robot.py
python tests/test_logger.py
python tests/test_robot.py
pip uninstall -y gym

echo "=================== Testing gym==0.26.2 ==================="
pip install gym==0.26.2
python tests/test_arms.py
python tests/test_all.py
pip uninstall -y gym

echo "=================== Testing gymnasium ==================="
pip install gymnasium
python tests/test_arms.py
python tests/test_all.py
pip uninstall -y gymnasium
