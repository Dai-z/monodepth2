cmake_minimum_required(VERSION 2.8.3)
project(depth)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  sensor_msgs
)

add_service_files(FILES Depth.srv)
generate_messages(
        DEPENDENCIES
        sensor_msgs
)