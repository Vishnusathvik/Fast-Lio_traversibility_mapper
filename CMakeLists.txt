cmake_minimum_required(VERSION 3.0.2)
project(fastlio_traversability_mapper)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs nav_msgs std_srvs message_filters)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS rospy sensor_msgs nav_msgs std_srvs message_filters)
catkin_install_python(PROGRAMS
  scripts/traversability_mapper_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
