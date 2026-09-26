#!/usr/bin/env python3
"""
Manda o robo andar a uma velocidade linear constante e imprime os valores do IMU.

Uso (depois de dar source ao ROS 2 e ter o gazebo.launch.py já a correr noutro
terminal):

    python3 mover_e_ler_imu.py --vel 0.2

    --vel       velocidade linear em m/s (default 0.2)
    --angular   velocidade angular em rad/s (default 0.0)
    --duracao   segundos a andar; 0 = anda para sempre até Ctrl+C (default 0)
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu


class MoverELerImu(Node):
    def __init__(self, vel_linear: float, vel_angular: float):
        super().__init__('mover_e_ler_imu')

        self.vel_linear = vel_linear
        self.vel_angular = vel_angular

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # IMPORTANTE: a bridge publica o IMU com QoS "best effort" (sensor data).
        # Uma subscrição "reliable" (o default do rclpy) não recebe nada dela.
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile_sensor_data
        )

        # Manda o comando de velocidade repetidamente (a bridge/plugin espera
        # receção contínua, um Twist só uma vez pode não bastar)
        self.cmd_timer = self.create_timer(0.05, self.publicar_cmd_vel)

    def publicar_cmd_vel(self):
        msg = Twist()
        msg.linear.x = self.vel_linear
        msg.angular.z = self.vel_angular
        self.cmd_pub.publish(msg)

    def imu_callback(self, msg: Imu):
        ax, ay, az = (msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z)
        gx, gy, gz = (msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z)
        self.get_logger().info(
            f'accel [m/s^2]: x={ax:+.3f} y={ay:+.3f} z={az:+.3f} | '
            f'gyro [rad/s]: x={gx:+.3f} y={gy:+.3f} z={gz:+.3f}'
        )

    def parar(self):
        msg = Twist()  # tudo a 0 -> travar o robo
        self.cmd_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vel', type=float, default=0.2, help='velocidade linear em m/s')
    parser.add_argument('--angular', type=float, default=0.0, help='velocidade angular em rad/s')
    parser.add_argument('--duracao', type=float, default=0.0,
                         help='segundos a andar (0 = infinito, para com Ctrl+C)')
    args, _ = parser.parse_known_args()

    rclpy.init(args=sys.argv)
    node = MoverELerImu(args.vel, args.angular)

    try:
        if args.duracao > 0:
            rclpy.spin_once(node, timeout_sec=0.0)  # garante que o publisher existe
            import time
            fim = time.time() + args.duracao
            while rclpy.ok() and time.time() < fim:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.parar()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()