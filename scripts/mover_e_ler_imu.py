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
import math
import threading
import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, MagneticField, JointState
from nav_msgs.msg import Odometry


def quat_para_euler_graus(x, y, z, w):
    """Converte quaternion (x,y,z,w) em (roll, pitch, heading) em graus.

    heading/yaw é normalizado para [0, 360), à semelhança do registo
    EUL_Heading do BNO055 real (formato Android: 0°-360°, aumenta ao
    rodar no sentido horário). Nota: como a fusão aqui vem do
    imu_filter_madgwick com world_frame='enu', o sentido de rotação do
    yaw é o convencional ENU (anti-horário), não o do compass real —
    serve para veres a orientação do robô no simulador, não como
    bússola magnética calibrada.
    """
    # roll (rotação em X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (rotação em Y)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))  # evita erro de arredondamento no asin
    pitch = math.asin(sinp)

    # heading / yaw (rotação em Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    heading = math.atan2(siny_cosp, cosy_cosp)

    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    heading_deg = math.degrees(heading) % 360.0

    return roll_deg, pitch_deg, heading_deg


GRAVIDADE = 9.8  # m/s^2, tem de bater certo com <gravity> no raf4_world.sdf


def rodar_vetor_por_quaternion(qx, qy, qz, qw, vx, vy, vz):
    """Roda o vetor (vx,vy,vz) do referencial do corpo (IMU) para o
    referencial do mundo, usando o quaternion de orientação fundida.
    Fórmula: v' = v + 2*qw*(u×v) + 2*(u×(u×v)), com u=(qx,qy,qz).
    """
    ux, uy, uz = qx, qy, qz
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    ccx = uy * cz - uz * cy
    ccy = uz * cx - ux * cz
    ccz = ux * cy - uy * cx
    return (vx + 2 * qw * cx + 2 * ccx,
            vy + 2 * qw * cy + 2 * ccy,
            vz + 2 * qw * cz + 2 * ccz)


# ============================================================
# Encoder dos motores WaveShare N20 (12V, 200RPM, 1:150) que usas.
# Especificação do fabricante (waveshare.com/wiki/DCGM-N20-12V-EN-200RPM):
#   - Hall Resolution: básico 7 PPR x redução 1:150 = 1050 PPR no veio
#     de saída (isto já é o pulso de UM canal, contado numa só aresta)
#   - Encoder AB de duas fases (quadratura) -> se o firmware contar as
#     4 transições por ciclo (subida/descida de A e B), a resolução
#     efetiva sobe 4x: 1050 * 4 = 4200 contagens/volta
# Aqui simulamos a contagem de quadratura completa (4x), que é o que a
# generalidade do firmware faz (ex: bibliotecas de encoder do Arduino/
# ESP32 com interrupções em ambos os canais) porque dá o dobro/quádruplo
# da resolução "de borla". Muda ENCODER_QUAD_X4 para False se o teu
# firmware só conta um canal/aresta.
# ============================================================
PPR_BASE = 7
REDUCAO_ENGRENAGEM = 150
ENCODER_QUAD_X4 = True
CONTAGENS_POR_VOLTA = PPR_BASE * REDUCAO_ENGRENAGEM * (4 if ENCODER_QUAD_X4 else 1)

# Têm de bater certo com o raf4.urdf.xacro (wheel_radius, wheel_separation
# do plugin DiffDrive) -- se mudares lá, muda aqui também.
RAIO_RODA = 0.0235       # m
DISTANCIA_ENTRE_RODAS = 0.166  # m


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
        # Odometria das rodas (encoders), publicada pelo plugin DiffDrive do
        # Gazebo — serve de "verdade de referência" para comparar com a
        # posição obtida por dupla integração do IMU.
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, qos_profile_sensor_data
        )
        # Ângulo "verdadeiro" (contínuo) de cada roda, publicado pelo
        # JointStatePublisher do Gazebo -> usado para simular os pulsos
        # reais do encoder Hall dos motores N20 (ver constantes acima).
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, qos_profile_sensor_data
        )

        self.ultimo_accel = (0.0, 0.0, 0.0)
        self.ultimo_gyro = (0.0, 0.0, 0.0)
        self.ultimo_mag = (0.0, 0.0, 0.0)
        self.ultima_orientacao = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w (identidade)

        # ---- Estado da integração da aceleração (dead-reckoning por IMU) ----
        # vel/pos em coordenadas do MUNDO (não do corpo do robô)
        self.imu_vel = [0.0, 0.0, 0.0]
        self.imu_pos = [0.0, 0.0, 0.0]
        self._t_imu_anterior = None  # timestamp (segundos, float) da última msg

        # ---- Odometria das rodas (referência, "perfeita" via física) ----
        self.odom_pos = (0.0, 0.0)
        self.odom_yaw_deg = 0.0

        # ---- Odometria calculada a partir dos encoders (quantizada) ----
        # x, y, theta (rad) -- pose estimada só com os pulsos dos encoders,
        # tal como o firmware real do robô faria
        self.enc_pose = [0.0, 0.0, 0.0]
        self.enc_ticks_esq = 0   # contagem acumulada (inteiro, como um
        self.enc_ticks_dir = 0   # contador de hardware real)
        self._angulo_anterior_esq = None  # último ângulo contínuo (rad) lido
        self._angulo_anterior_dir = None  # do joint_states, p/ calcular delta

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
        # orientação já fundida pelo imu_filter_madgwick; o campo
        # linear_acceleration vem re-publicado sem alteração a partir de
        # imu/data_raw, por isso dá para integrar aqui mesmo, já
        # sincronizado com a orientação que lhe corresponde.
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._lock:
            self.ultima_orientacao = (qx, qy, qz, qw)

            if self._t_imu_anterior is None:
                self._t_imu_anterior = t_msg
                return
            dt = t_msg - self._t_imu_anterior
            self._t_imu_anterior = t_msg
            if dt <= 0.0 or dt > 0.5:
                # relógio voltou atrás ou passou-se demasiado tempo (ex:
                # simulação pausada) -> não integra esta amostra, evita
                # um "salto" artificial na posição
                return

            # aceleração medida no referencial do CORPO (accel "crua", tal
            # como um acelerómetro real: inclui a reação à gravidade)
            ax_b = msg.linear_acceleration.x
            ay_b = msg.linear_acceleration.y
            az_b = msg.linear_acceleration.z

            # roda para o referencial do MUNDO usando a orientação fundida
            ax_w, ay_w, az_w = rodar_vetor_por_quaternion(qx, qy, qz, qw, ax_b, ay_b, az_b)

            # remove a gravidade -> fica só a aceleração de translação real
            # (a gravidade mundial é (0,0,-g); o acelerómetro em repouso
            # mede a reação a isso, +g no eixo Z do mundo, por isso subtrai-se)
            az_w -= GRAVIDADE

            # integração dupla (Euler simples): a -> v -> x
            self.imu_vel[0] += ax_w * dt
            self.imu_vel[1] += ay_w * dt
            self.imu_vel[2] += az_w * dt

            self.imu_pos[0] += self.imu_vel[0] * dt
            self.imu_pos[1] += self.imu_vel[1] * dt
            self.imu_pos[2] += self.imu_vel[2] * dt

    def odom_callback(self, msg: Odometry):
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        # yaw a partir de um quaternion "plano" (só rotação em Z, é o caso
        # da odometria de um robô diferencial) -> fórmula simplificada
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        with self._lock:
            self.odom_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            self.odom_yaw_deg = math.degrees(yaw) % 360.0

    def joint_callback(self, msg: JointState):
        """Simula os pulsos reais do encoder Hall dos motores N20 e
        recalcula a odometria a partir deles, exatamente como o firmware
        de um robô real faria (contar pulsos -> distância -> pose)."""
        try:
            i_esq = msg.name.index('left_wheel_joint')
            i_dir = msg.name.index('right_wheel_joint')
        except ValueError:
            return  # mensagem ainda não tem os nomes esperados
        ang_esq = msg.position[i_esq]  # rad, ângulo contínuo "verdadeiro"
        ang_dir = msg.position[i_dir]

        with self._lock:
            if self._angulo_anterior_esq is None:
                # primeira mensagem: só inicializa, não há delta para integrar
                self._angulo_anterior_esq = ang_esq
                self._angulo_anterior_dir = ang_dir
                return

            # ---- 1) Quantização: ângulo contínuo -> nº de pulsos (inteiro) ----
            # Isto é o que um contador de hardware real faz: só existe em
            # incrementos de 1/CONTAGENS_POR_VOLTA de volta, nunca frações.
            ticks_esq_novo = round(ang_esq / (2 * math.pi) * CONTAGENS_POR_VOLTA)
            ticks_dir_novo = round(ang_dir / (2 * math.pi) * CONTAGENS_POR_VOLTA)
            delta_ticks_esq = ticks_esq_novo - self.enc_ticks_esq
            delta_ticks_dir = ticks_dir_novo - self.enc_ticks_dir
            self.enc_ticks_esq = ticks_esq_novo
            self.enc_ticks_dir = ticks_dir_novo

            self._angulo_anterior_esq = ang_esq
            self._angulo_anterior_dir = ang_dir

            if delta_ticks_esq == 0 and delta_ticks_dir == 0:
                return  # roda não rodou o suficiente para um novo pulso

            # ---- 2) Pulsos -> distância percorrida por cada roda ----
            perimetro = 2 * math.pi * RAIO_RODA
            d_esq = (delta_ticks_esq / CONTAGENS_POR_VOLTA) * perimetro
            d_dir = (delta_ticks_dir / CONTAGENS_POR_VOLTA) * perimetro

            # ---- 3) Cinemática de robô diferencial (odometria clássica) ----
            d_centro = (d_esq + d_dir) / 2.0
            d_theta = (d_dir - d_esq) / DISTANCIA_ENTRE_RODAS

            x, y, theta = self.enc_pose
            theta_meio = theta + d_theta / 2.0
            x += d_centro * math.cos(theta_meio)
            y += d_centro * math.sin(theta_meio)
            theta = (theta + d_theta + math.pi) % (2 * math.pi) - math.pi
            self.enc_pose = [x, y, theta]

    def reset_odometria_imu(self):
        """Zera a posição/velocidade estimadas por integração do IMU.
        Útil porque o drift torna a estimativa inútil ao fim de pouco
        tempo — não há problema nenhum em "recomeçar a contagem"."""
        with self._lock:
            self.imu_vel = [0.0, 0.0, 0.0]
            self.imu_pos = [0.0, 0.0, 0.0]
            self._t_imu_anterior = None

    def get_leituras(self):
        """Chamado pela GUI: devolve uma cópia consistente das leituras."""
        with self._lock:
            return (self.ultimo_accel, self.ultimo_gyro,
                    self.ultimo_mag, self.ultima_orientacao,
                    tuple(self.imu_vel), tuple(self.imu_pos),
                    self.odom_pos, self.odom_yaw_deg,
                    tuple(self.enc_pose), self.enc_ticks_esq, self.enc_ticks_dir)

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
        self.geometry('480x760')
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

        self.btn_reset_imu = ttk.Button(
            frame_botoes, text='↺ Reset odom. IMU', command=self._on_reset_imu
        )
        self.btn_reset_imu.pack(side='left', padx=4)

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
        self.var_heading = tk.StringVar(value='Heading: 0.0°   Roll: 0.0°   Pitch: 0.0°')

        for var in (self.var_accel, self.var_gyro, self.var_mag, self.var_orient):
            ttk.Label(frame_imu, textvariable=var, font=('Courier', 11)).pack(anchor='w', **pad)

        # Heading/roll/pitch em destaque (é o que se lê mais facilmente,
        # equivalente ao EUL_Heading/Roll/Pitch do BNO055 real)
        ttk.Separator(frame_imu, orient='horizontal').pack(fill='x', padx=8, pady=2)
        ttk.Label(
            frame_imu, textvariable=self.var_heading,
            font=('Courier', 12, 'bold'), foreground='#1a5aa8'
        ).pack(anchor='w', **pad)

        # --- Velocidade/posição por dupla integração do IMU (com aviso) ---
        ttk.Separator(frame_imu, orient='horizontal').pack(fill='x', padx=8, pady=2)
        ttk.Label(
            frame_imu, text='Dead-reckoning por IMU (deriva com o tempo!):',
            font=('Courier', 9, 'italic'), foreground='#994400'
        ).pack(anchor='w', padx=8)

        self.var_imu_vel = tk.StringVar(value='vel: (0.000, 0.000, 0.000) m/s')
        self.var_imu_pos = tk.StringVar(value='pos: (0.000, 0.000, 0.000) m')
        for var in (self.var_imu_vel, self.var_imu_pos):
            ttk.Label(frame_imu, textvariable=var, font=('Courier', 11)).pack(anchor='w', **pad)

        # --- Odometria calculada a partir dos encoders reais (N20) ---
        ttk.Separator(frame_imu, orient='horizontal').pack(fill='x', padx=8, pady=2)
        ttk.Label(
            frame_imu,
            text=f'Odometria por encoders ({CONTAGENS_POR_VOLTA} contagens/volta):',
            font=('Courier', 9, 'italic'), foreground='#7a3ba0'
        ).pack(anchor='w', padx=8)
        self.var_enc_pose = tk.StringVar(value='pos: (0.000, 0.000) m   yaw: 0.0°')
        self.var_enc_ticks = tk.StringVar(value='ticks: esq=0  dir=0')
        ttk.Label(frame_imu, textvariable=self.var_enc_pose, font=('Courier', 11)).pack(anchor='w', **pad)
        ttk.Label(frame_imu, textvariable=self.var_enc_ticks, font=('Courier', 11)).pack(anchor='w', **pad)

        # --- Odometria das rodas, para comparação (muito mais fiável) ---
        ttk.Separator(frame_imu, orient='horizontal').pack(fill='x', padx=8, pady=2)
        ttk.Label(
            frame_imu, text='Odometria "perfeita" do Gazebo (referência):',
            font=('Courier', 9, 'italic'), foreground='#227722'
        ).pack(anchor='w', padx=8)
        self.var_odom = tk.StringVar(value='pos: (0.000, 0.000) m   yaw: 0.0°')
        ttk.Label(frame_imu, textvariable=self.var_odom, font=('Courier', 11)).pack(anchor='w', **pad)

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

    def _on_reset_imu(self):
        self.node.reset_odometria_imu()

    def _atualizar_leituras(self):
        (accel, gyro, mag, orient,
         imu_vel, imu_pos, odom_pos, odom_yaw_deg,
         enc_pose, enc_ticks_esq, enc_ticks_dir) = self.node.get_leituras()
        ax, ay, az = accel
        gx, gy, gz = gyro
        mx, my, mz = mag
        qx, qy, qz, qw = orient
        vx, vy, vz = imu_vel
        px, py, pz = imu_pos
        ox, oy = odom_pos
        ex, ey, etheta = enc_pose

        self.var_accel.set(f'accel: ({ax:+.3f}, {ay:+.3f}, {az:+.3f}) m/s²')
        self.var_gyro.set(f'gyro: ({gx:+.3f}, {gy:+.3f}, {gz:+.3f}) rad/s')
        self.var_mag.set(f'mag: ({mx:+.6f}, {my:+.6f}, {mz:+.6f}) T')
        self.var_orient.set(f'orient (quat): ({qx:+.3f}, {qy:+.3f}, {qz:+.3f}, {qw:+.3f})')

        roll_deg, pitch_deg, heading_deg = quat_para_euler_graus(qx, qy, qz, qw)
        self.var_heading.set(
            f'Heading: {heading_deg:6.1f}°   Roll: {roll_deg:+6.1f}°   Pitch: {pitch_deg:+6.1f}°'
        )

        self.var_imu_vel.set(f'vel: ({vx:+.3f}, {vy:+.3f}, {vz:+.3f}) m/s')
        self.var_imu_pos.set(f'pos: ({px:+.3f}, {py:+.3f}, {pz:+.3f}) m')
        self.var_enc_pose.set(
            f'pos: ({ex:+.3f}, {ey:+.3f}) m   yaw: {math.degrees(etheta) % 360.0:5.1f}°'
        )
        self.var_enc_ticks.set(f'ticks: esq={enc_ticks_esq}  dir={enc_ticks_dir}')
        self.var_odom.set(f'pos: ({ox:+.3f}, {oy:+.3f}) m   yaw: {odom_yaw_deg:5.1f}°')

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