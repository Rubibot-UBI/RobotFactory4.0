#!/usr/bin/env python3
"""
Interface gráfica para mandar o robô andar e ver os valores do IMU (accel,
gyro, magnetómetro e orientação fundida) a atualizar em tempo real.

Uso (depois de dar source ao ROS 2 e ter o gazebo.launch.py já a correr
noutro terminal):

    python3 mover_e_ler_imu_gui.py

Se o "import tkinter" falhar, instala com:
    sudo apt install python3-tk
"""

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, MagneticField


class MoverELerImuNode(Node):
    """Nó ROS 2: publica cmd_vel e guarda as últimas leituras do IMU.

    Corre numa thread própria (rclpy.spin); a GUI só lê o estado através
    de get_leituras()/esta_a_andar(), sempre protegido por um lock.
    """

    def __init__(self):
        super().__init__('mover_e_ler_imu_gui')

        self._lock = threading.Lock()
        self.vel_linear = 0.0
        self.vel_angular = 0.0
        self._a_andar = False
        self._parar_em = None  # timestamp (time.time()); None = contínuo

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # IMPORTANTE: a bridge publica o IMU com QoS "best effort" (sensor
        # data). Uma subscrição "reliable" (o default do rclpy) não recebe
        # nada dela.
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data_raw', self.imu_callback, qos_profile_sensor_data
        )
        self.mag_sub = self.create_subscription(
            MagneticField, 'imu/mag', self.mag_callback, qos_profile_sensor_data
        )
        self.fused_sub = self.create_subscription(
            Imu, 'imu/data', self.fused_callback, qos_profile_sensor_data
        )

        self.ultimo_accel = (0.0, 0.0, 0.0)
        self.ultimo_gyro = (0.0, 0.0, 0.0)
        self.ultimo_mag = (0.0, 0.0, 0.0)
        self.ultima_orientacao = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w (identidade)

        # Um único timer trata de publicar o cmd_vel repetidamente (a
        # bridge/plugin espera receção contínua) e de verificar se a
        # duração pedida já terminou. Não cria timers a partir da thread
        # da GUI — só este, criado aqui na própria thread do nó.
        self.create_timer(0.05, self._tick)

    # ---------- Callbacks dos sensores (correm na thread do rclpy.spin) ----------
    def imu_callback(self, msg: Imu):
        with self._lock:
            self.ultimo_accel = (msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z)
            self.ultimo_gyro = (msg.angular_velocity.x,
                                 msg.angular_velocity.y,
                                 msg.angular_velocity.z)

    def mag_callback(self, msg: MagneticField):
        with self._lock:
            self.ultimo_mag = (msg.magnetic_field.x,
                                msg.magnetic_field.y,
                                msg.magnetic_field.z)

    def fused_callback(self, msg: Imu):
        # orientação já fundida pelo imu_filter_madgwick
        with self._lock:
            self.ultima_orientacao = (msg.orientation.x,
                                       msg.orientation.y,
                                       msg.orientation.z,
                                       msg.orientation.w)

    def get_leituras(self):
        """Chamado pela GUI: devolve uma cópia consistente das leituras."""
        with self._lock:
            return (self.ultimo_accel, self.ultimo_gyro,
                    self.ultimo_mag, self.ultima_orientacao)

    def esta_a_andar(self):
        with self._lock:
            return self._a_andar

    # ---------- Controlo de movimento (chamado pela GUI) ----------
    def _tick(self):
        with self._lock:
            a_andar = self._a_andar
            vl, va = self.vel_linear, self.vel_angular
            parar_em = self._parar_em

        # duração pedida já terminou -> pára sozinho
        if a_andar and parar_em is not None and time.time() >= parar_em:
            a_andar = False
            with self._lock:
                self._a_andar = False
                self._parar_em = None

        msg = Twist()
        if a_andar:
            msg.linear.x = vl
            msg.angular.z = va
        self.cmd_pub.publish(msg)  # publica sempre; 0 quando parado -> trava o robô

    def andar(self, vel_linear: float, vel_angular: float, duracao: float):
        with self._lock:
            self.vel_linear = vel_linear
            self.vel_angular = vel_angular
            self._a_andar = True
            self._parar_em = (time.time() + duracao) if duracao > 0 else None

    def parar(self):
        with self._lock:
            self._a_andar = False
            self.vel_linear = 0.0
            self.vel_angular = 0.0
            self._parar_em = None


class App(tk.Tk):
    def __init__(self, node: MoverELerImuNode):
        super().__init__()
        self.node = node
        self.title('Mover e Ler IMU — RobotFactory4.0')
        self.geometry('440x420')
        self.resizable(False, False)

        self._construir_widgets()
        self._atualizar_leituras()  # arranca o loop de atualização a 10 Hz

    def _construir_widgets(self):
        pad = {'padx': 8, 'pady': 4}

        # --- Controlo de movimento ---
        frame_ctrl = ttk.LabelFrame(self, text='Controlo')
        frame_ctrl.pack(fill='x', **pad)

        ttk.Label(frame_ctrl, text='Velocidade linear (m/s):').grid(row=0, column=0, sticky='w', **pad)
        self.entry_vel = ttk.Entry(frame_ctrl, width=10)
        self.entry_vel.insert(0, '0.2')
        self.entry_vel.grid(row=0, column=1, **pad)

        ttk.Label(frame_ctrl, text='Velocidade angular (rad/s):').grid(row=1, column=0, sticky='w', **pad)
        self.entry_angular = ttk.Entry(frame_ctrl, width=10)
        self.entry_angular.insert(0, '0.0')
        self.entry_angular.grid(row=1, column=1, **pad)

        ttk.Label(frame_ctrl, text='Duração (s, 0 = contínuo):').grid(row=2, column=0, sticky='w', **pad)
        self.entry_duracao = ttk.Entry(frame_ctrl, width=10)
        self.entry_duracao.insert(0, '0')
        self.entry_duracao.grid(row=2, column=1, **pad)

        frame_botoes = ttk.Frame(frame_ctrl)
        frame_botoes.grid(row=3, column=0, columnspan=2, pady=6)

        self.btn_andar = ttk.Button(frame_botoes, text='▶ Andar', command=self._on_andar)
        self.btn_andar.pack(side='left', padx=4)

        self.btn_parar = ttk.Button(frame_botoes, text='■ Parar', command=self._on_parar)
        self.btn_parar.pack(side='left', padx=4)

        self.status_var = tk.StringVar(value='Parado')
        ttk.Label(frame_ctrl, textvariable=self.status_var, foreground='blue').grid(
            row=4, column=0, columnspan=2, **pad
        )

        # --- Leituras do IMU ---
        frame_imu = ttk.LabelFrame(self, text='IMU (tempo real)')
        frame_imu.pack(fill='both', expand=True, **pad)

        self.var_accel = tk.StringVar(value='accel: (0.000, 0.000, 0.000) m/s²')
        self.var_gyro = tk.StringVar(value='gyro: (0.000, 0.000, 0.000) rad/s')
        self.var_mag = tk.StringVar(value='mag: (0.000000, 0.000000, 0.000000) T')
        self.var_orient = tk.StringVar(value='orient (quat): (0.000, 0.000, 0.000, 1.000)')

        for var in (self.var_accel, self.var_gyro, self.var_mag, self.var_orient):
            ttk.Label(frame_imu, textvariable=var, font=('Courier', 11)).pack(anchor='w', **pad)

    def _on_andar(self):
        try:
            vel = float(self.entry_vel.get())
            angular = float(self.entry_angular.get())
            duracao = float(self.entry_duracao.get())
        except ValueError:
            self.status_var.set('Valores inválidos!')
            return

        self.node.andar(vel, angular, duracao)
        self.status_var.set(f'A andar {duracao:.1f}s...' if duracao > 0 else 'A andar (contínuo)...')

    def _on_parar(self):
        self.node.parar()
        self.status_var.set('Parado')

    def _atualizar_leituras(self):
        accel, gyro, mag, orient = self.node.get_leituras()
        ax, ay, az = accel
        gx, gy, gz = gyro
        mx, my, mz = mag
        qx, qy, qz, qw = orient

        self.var_accel.set(f'accel: ({ax:+.3f}, {ay:+.3f}, {az:+.3f}) m/s²')
        self.var_gyro.set(f'gyro: ({gx:+.3f}, {gy:+.3f}, {gz:+.3f}) rad/s')
        self.var_mag.set(f'mag: ({mx:+.6f}, {my:+.6f}, {mz:+.6f}) T')
        self.var_orient.set(f'orient (quat): ({qx:+.3f}, {qy:+.3f}, {qz:+.3f}, {qw:+.3f})')

        # se a duração terminou sozinha (timer do nó), refletir isso no status
        if not self.node.esta_a_andar() and self.status_var.get().startswith('A andar'):
            self.status_var.set('Parado (duração terminada)')

        self.after(100, self._atualizar_leituras)  # 10 Hz

    def on_close(self):
        self.node.parar()
        self.destroy()


def _ros_spin_thread(node):
    rclpy.spin(node)


def main():
    rclpy.init(args=sys.argv)
    node = MoverELerImuNode()

    # rclpy.spin corre numa thread à parte para não bloquear o mainloop do Tkinter
    spin_thread = threading.Thread(target=_ros_spin_thread, args=(node,), daemon=True)
    spin_thread.start()

    app = App(node)
    app.protocol('WM_DELETE_WINDOW', app.on_close)
    try:
        app.mainloop()
    finally:
        node.parar()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()