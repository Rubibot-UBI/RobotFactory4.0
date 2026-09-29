#!/usr/bin/env python3
"""
Interface gráfica para mandar o robô andar e ver os valores do IMU (accel,
gyro, magnetómetro e orientação fundida) a atualizar em tempo real.

Uso (depois de dar source ao ROS 2 e ter o gazebo.launch.py já a correr
noutro terminal):

    python3 mover_e_ler_imu.py

Se o "import tkinter" falhar, instala com:
    sudo apt install python3-tk

O botão "Gráficos" abre uma janela com aceleração, velocidade e posição
(IMU por dupla integração vs odometria do Gazebo). Precisa de matplotlib:
    sudo apt install python3-matplotlib      (ou: pip install matplotlib)
"""

import sys
import time
import math
import threading
import tkinter as tk
from tkinter import ttk

try:
    import numpy as np
except ImportError:      # só os gráficos precisam de numpy
    np = None

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
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
DISTANCIA_ENTRE_RODAS = 0.16595  # m (= left_wheel_y - right_wheel_y no xacro)


class MoverELerImuNode(Node):
    """Nó ROS 2: publica cmd_vel e guarda as últimas leituras do IMU.

    Corre numa thread própria (rclpy.spin); a GUI só lê o estado através
    de get_leituras()/esta_a_andar(), sempre protegido por um lock.
    """

    def __init__(self):
        # use_sim_time: todos os tempos (duração do movimento, dt do IMU)
        # passam a ser tempo de SIMULAÇÃO, coerentes entre si mesmo com RTF<1.
        # Precisa do /clock bridgeado (já está no bridge.yaml).
        super().__init__(
            'mover_e_ler_imu_gui',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self._lock = threading.Lock()
        self.vel_linear = 0.0
        self.vel_angular = 0.0
        self._a_andar = False
        self._parar_em = None  # tempo de simulação (s); None = contínuo

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

        # ---- Histórico para os gráficos (tempo em s desde o início/reset) ----
        # Buffers em anel de numpy (escrita O(1), leitura da cauda sem loops
        # Python). IMU: [t, ax_w, ay_w, vx, vy, px, py]  (aceleração no mundo,
        # sem gravidade). Odom: [t, |v|, distância à origem]  (referência).
        if np is not None:
            self._buf_imu = np.zeros((60000, 7))    # ~10 min a 100 Hz
            self._buf_odom = np.zeros((30000, 3))   # ~10 min a 50 Hz
        self._n_imu = 0    # nº total de amostras escritas (também serve de "versão")
        self._n_odom = 0
        self._hist_t0 = None
        self._odom_p0 = None

        # ---- Odometria das rodas (referência, "perfeita" via física) ----
        self.odom_pos = (0.0, 0.0)
        self.odom_yaw_deg = 0.0

        # ---- Odometria calculada a partir dos encoders (quantizada) ----
        # x, y, theta (rad) -- pose estimada só com os pulsos dos encoders,
        # tal como o firmware real do robô faria
        self.enc_pose = [0.0, 0.0, 0.0]
        self._enc_inicializado = False
        self.enc_ticks_esq = 0   # contagem acumulada (inteiro, como um
        self.enc_ticks_dir = 0   # contador de hardware real)

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
                if self._hist_t0 is None:
                    self._hist_t0 = t_msg
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

            if self._hist_t0 is None:
                self._hist_t0 = t_msg
            if np is not None:
                self._buf_imu[self._n_imu % len(self._buf_imu)] = (
                    t_msg - self._hist_t0, ax_w, ay_w,
                    self.imu_vel[0], self.imu_vel[1],
                    self.imu_pos[0], self.imu_pos[1])
                self._n_imu += 1

    def odom_callback(self, msg: Odometry):
        o = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y),
                         1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        px, py = msg.pose.pose.position.x, msg.pose.pose.position.y
        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        vel = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        with self._lock:
            self.odom_pos = (px, py)
            self.odom_yaw_deg = math.degrees(yaw) % 360.0
            # histórico para os gráficos (referência)
            if self._odom_p0 is None:
                self._odom_p0 = (px, py)
            if self._hist_t0 is None:
                self._hist_t0 = t_msg
            if np is not None:
                self._buf_odom[self._n_odom % len(self._buf_odom)] = (
                    t_msg - self._hist_t0, vel,
                    math.hypot(px - self._odom_p0[0], py - self._odom_p0[1]))
                self._n_odom += 1

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
            if self._enc_inicializado is False:
                # primeira mensagem: só inicializa, não há delta para integrar
                # sem isto, se a roda já estiver rodada (GUI aberta depois do
                # robô ter andado), o 1.º delta seria enorme -> "salto" na pose
                self.enc_ticks_esq = round(ang_esq / (2 * math.pi) * CONTAGENS_POR_VOLTA)
                self.enc_ticks_dir = round(ang_dir / (2 * math.pi) * CONTAGENS_POR_VOLTA)
                self._enc_inicializado = True
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
            self._n_imu = 0
            self._n_odom = 0
            self._hist_t0 = None
            self._odom_p0 = None

    @staticmethod
    def _cauda(buf, n, t_min):
        """Linhas do buffer em anel com t >= t_min, por ordem temporal.
        Só copia a cauda necessária (dobra o bloco até cobrir a janela)."""
        cap = len(buf)
        total = min(n, cap)
        if total == 0:
            return buf[:0]
        m = min(total, 512)
        while True:
            bloco = buf[np.arange(n - m, n) % cap]
            if m >= total or bloco[0, 0] < t_min:
                break
            m = min(total, m * 2)
        return bloco[bloco[:, 0] >= t_min] if bloco[0, 0] < t_min else bloco

    def versao_historico(self):
        """Muda sempre que chega uma amostra nova (evita redesenhos inúteis)."""
        return self._n_imu + self._n_odom

    def get_historico(self, janela=30.0):
        """(imu[k,7], odom[k,3], t_fim) só com os últimos `janela` s (0 = tudo)."""
        with self._lock:
            t_fim = 0.0
            if self._n_imu:
                t_fim = max(t_fim, self._buf_imu[(self._n_imu - 1) % len(self._buf_imu), 0])
            if self._n_odom:
                t_fim = max(t_fim, self._buf_odom[(self._n_odom - 1) % len(self._buf_odom), 0])
            t_min = (t_fim - janela) if janela > 0 else -1.0
            return (self._cauda(self._buf_imu, self._n_imu, t_min),
                    self._cauda(self._buf_odom, self._n_odom, t_min), t_fim)

    def get_leituras(self):
        """Chamado pela GUI: devolve uma cópia consistente das leituras."""
        with self._lock:
            return (self.ultimo_accel, self.ultimo_gyro,
                    self.ultimo_mag, self.ultima_orientacao,
                    tuple(self.imu_vel), tuple(self.imu_pos),
                    self.odom_pos, self.odom_yaw_deg,
                    tuple(self.enc_pose), self.enc_ticks_esq, self.enc_ticks_dir)

    def _agora(self):
        """Tempo (s) do relógio do nó = tempo de simulação (use_sim_time)."""
        return self.get_clock().now().nanoseconds * 1e-9

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
        if a_andar and parar_em is not None and self._agora() >= parar_em:
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
            self._parar_em = (self._agora() + duracao) if duracao > 0 else None

    def parar(self):
        with self._lock:
            self._a_andar = False
            self.vel_linear = 0.0
            self.vel_angular = 0.0
            self._parar_em = None


class JanelaGraficos(tk.Toplevel):
    """Janela com 3 gráficos EM TEMPO REAL: aceleração, velocidade, posição.

    IMU (dead-reckoning) a cheio; odometria do Gazebo a tracejado verde.
    Comparação em normas (|v| e distância): robusta a diferenças de yaw
    entre o referencial do IMU (Madgwick + magnetómetro, ENU) e o do "odom"
    (o robô nasce virado para +X = Este, por isso devem coincidir a menos
    do erro do magnetómetro/bias simulado).

    Eficiência:
      * o eixo X é "tempo relativo a agora" (-janela..0), logo os eixos não
        mudam entre frames e só as linhas são redesenhadas (blitting);
      * o eixo Y só muda quando os dados saem dele (ou sobra muito espaço),
        e só então há redesenho completo;
      * não se redesenha se não chegaram amostras novas;
      * buffers numpy, só a cauda visível é lida, decimação a MAX_PONTOS.
    """

    ALVO_MS = 40          # ~25 fps no máximo
    MAX_PONTOS = 800      # por linha; acima disto faz-se decimação

    def __init__(self, master, node: MoverELerImuNode):
        if np is None:
            raise ImportError('numpy')
        super().__init__(master)
        self.node = node
        self.title('Gráficos em tempo real — aceleração, velocidade e posição')
        self.geometry('900x800')
        self._ativa = True
        self._pausado = False
        self._fundo = None            # fundo em cache para o blitting
        self._versao = -1
        self._janela_ant = None
        self._ylim_t = {}             # última alteração de cada eixo Y

        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        barra = ttk.Frame(self)
        barra.pack(fill='x', padx=8, pady=4)
        ttk.Label(barra, text='Janela (s, 0 = tudo):').pack(side='left')
        self.entry_janela = ttk.Entry(barra, width=6)
        self.entry_janela.insert(0, '30')
        self.entry_janela.pack(side='left', padx=4)
        self.btn_pausa = ttk.Button(barra, text='⏸ Pausar', command=self._toggle_pausa)
        self.btn_pausa.pack(side='left', padx=4)
        ttk.Button(barra, text='↺ Reset (IMU + gráficos)',
                   command=self.node.reset_odometria_imu).pack(side='left', padx=4)
        self.var_info = tk.StringVar(value='à espera de dados do IMU…')
        ttk.Label(barra, textvariable=self.var_info, font=('Courier', 9),
                  foreground='#227722').pack(side='left', padx=10)

        self.fig = Figure(figsize=(9, 7.5), dpi=100)
        self.ax_a, self.ax_v, self.ax_p = self.fig.subplots(3, 1, sharex=True)
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.07, hspace=0.12)

        def linha(ax, **kw):   # animated=True -> só desenhada no blit
            l, = ax.plot([], [], animated=True, **kw)
            return l
        L = self.linhas = {}
        L['ax'] = linha(self.ax_a, color='#d62728', lw=1, label='a$_x$ (mundo)')
        L['ay'] = linha(self.ax_a, color='#1f77b4', lw=1, label='a$_y$ (mundo)')
        self.ax_a.set_ylabel('Aceleração (m/s²)')

        L['vx'] = linha(self.ax_v, color='#d62728', lw=1, alpha=.6, label='v$_x$')
        L['vy'] = linha(self.ax_v, color='#1f77b4', lw=1, alpha=.6, label='v$_y$')
        L['vn'] = linha(self.ax_v, color='k', lw=2, label='|v| IMU')
        L['ov'] = linha(self.ax_v, color='#2ca02c', lw=2, ls='--', label='|v| odom (ref.)')
        self.ax_v.set_ylabel('Velocidade (m/s)')

        L['px'] = linha(self.ax_p, color='#d62728', lw=1, alpha=.6, label='x')
        L['py'] = linha(self.ax_p, color='#1f77b4', lw=1, alpha=.6, label='y')
        L['pn'] = linha(self.ax_p, color='k', lw=2, label='distância IMU')
        L['od'] = linha(self.ax_p, color='#2ca02c', lw=2, ls='--', label='distância odom (ref.)')
        self.ax_p.set_ylabel('Posição (m)')
        self.ax_p.set_xlabel('Tempo (s)  —  0 = agora')

        for ax in (self.ax_a, self.ax_v, self.ax_p):
            ax.grid(alpha=.3)
            ax.legend(loc='upper left', fontsize=8, ncol=4)
            ax.axhline(0, color='gray', lw=.5)
            ax.set_ylim(-0.05, 0.05)
        self.ax_p.set_xlim(-30, 0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        # sempre que há um desenho completo (resize, mudança de eixos...),
        # guarda o fundo e desenha por cima as linhas animadas
        self.canvas.mpl_connect('draw_event', self._on_draw)

        self._t_taxa = time.time()
        self._n_taxa = 0
        self._fps = 0.0

        self.protocol('WM_DELETE_WINDOW', self._fechar)
        self.canvas.draw()
        self._atualizar()

    # ---------- blitting ----------
    def _on_draw(self, _evento):
        self._fundo = self.canvas.copy_from_bbox(self.fig.bbox)
        for l in self.linhas.values():
            self.fig.draw_artist(l)

    def _blit(self):
        if self._fundo is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self._fundo)
        for l in self.linhas.values():
            l.axes.draw_artist(l)
        self.canvas.blit(self.fig.bbox)

    def _toggle_pausa(self):
        self._pausado = not self._pausado
        self.btn_pausa.config(text='▶ Continuar' if self._pausado else '⏸ Pausar')

    def _fechar(self):
        self._ativa = False
        self.destroy()

    # ---------- eixo Y com histerese ----------
    def _ajustar_y(self, ax, arrays, min_span=0.05):
        """True se mudou os limites (=> precisa de redesenho completo)."""
        lo = hi = 0.0
        for a in arrays:
            if len(a):
                lo = min(lo, float(a.min())); hi = max(hi, float(a.max()))
        span = max(hi - lo, min_span)
        m = span * 0.12
        novo = (lo - m, lo + span + m)
        atual = ax.get_ylim()
        agora = time.time()
        sai = novo[0] < atual[0] or novo[1] > atual[1]
        sobra = (atual[1] - atual[0]) > 3.0 * (novo[1] - novo[0]) and \
            agora - self._ylim_t.get(ax, 0.0) > 2.0
        if sai or sobra:
            pad = span * 0.25 if sai else span * 0.12   # folga extra ao expandir
            ax.set_ylim(lo - pad, lo + span + pad)
            self._ylim_t[ax] = agora
            return True
        return False

    # ---------- ciclo de atualização ----------
    def _atualizar(self):
        if not self._ativa:
            return
        t_ini = time.perf_counter()
        try:
            janela = float(self.entry_janela.get())
        except ValueError:
            janela = 30.0
        versao = self.node.versao_historico()
        mudou_janela = janela != self._janela_ant
        # sem amostras novas (ou em pausa) não há nada a fazer
        if not self._pausado and (versao != self._versao or mudou_janela):
            self._versao = versao
            self._janela_ant = janela
            self._desenhar(janela, mudou_janela)
        gasto = (time.perf_counter() - t_ini) * 1000.0
        self.after(max(5, int(self.ALVO_MS - gasto)), self._atualizar)

    def _dec(self, a):
        n = len(a)
        return a if n <= self.MAX_PONTOS else a[::n // self.MAX_PONTOS + 1]

    def _desenhar(self, janela, mudou_janela):
        imu, odom, t_fim = self.node.get_historico(janela)
        L = self.linhas
        completo = mudou_janela

        if len(imu):
            imu = self._dec(imu)
            t = imu[:, 0] - t_fim                    # tempo relativo a "agora"
            ax_w, ay_w = imu[:, 1], imu[:, 2]
            vx, vy = imu[:, 3], imu[:, 4]
            px, py = imu[:, 5], imu[:, 6]
            vn, pn = np.hypot(vx, vy), np.hypot(px, py)
            L['ax'].set_data(t, ax_w); L['ay'].set_data(t, ay_w)
            L['vx'].set_data(t, vx); L['vy'].set_data(t, vy); L['vn'].set_data(t, vn)
            L['px'].set_data(t, px); L['py'].set_data(t, py); L['pn'].set_data(t, pn)
        else:
            ax_w = ay_w = vx = vy = vn = px = py = pn = np.zeros(0)

        if len(odom):
            odom = self._dec(odom)
            to = odom[:, 0] - t_fim
            ov, od = odom[:, 1], odom[:, 2]
            L['ov'].set_data(to, ov); L['od'].set_data(to, od)
        else:
            ov = od = np.zeros(0)

        # eixo X fixo (só muda se o utilizador mudar a janela)
        if janela > 0:
            if mudou_janela:
                self.ax_p.set_xlim(-janela, 0)
        else:   # "tudo": o intervalo cresce -> redesenho completo
            self.ax_p.set_xlim(-max(t_fim, 1.0), 0)
            completo = True

        completo |= self._ajustar_y(self.ax_a, (ax_w, ay_w))
        completo |= self._ajustar_y(self.ax_v, (vx, vy, vn, ov))
        completo |= self._ajustar_y(self.ax_p, (px, py, pn, od))

        if completo:
            self.canvas.draw()      # redesenho completo (dispara _on_draw)
        else:
            self._blit()            # rápido: só as linhas

        # indicador de vida (taxa real de redesenho), atualizado 1x/s
        self._n_taxa += 1
        agora = time.time()
        if agora - self._t_taxa >= 1.0:
            self._fps = self._n_taxa / (agora - self._t_taxa)
            self._n_taxa = 0
            self._t_taxa = agora
            if len(imu):
                self.var_info.set(f'●  t={t_fim:6.1f}s   {self._fps:.0f} fps')
            else:
                self.var_info.set('à espera de dados do IMU… (o gazebo.launch.py está a correr?)')


class App(tk.Tk):
    def __init__(self, node: MoverELerImuNode):
        super().__init__()
        self.node = node
        self.title('Mover e Ler IMU — RobotFactory4.0')
        self.geometry('480x800')
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

        self.btn_graficos = ttk.Button(
            frame_ctrl, text='📈 Gráficos (a, v, pos)', command=self._on_graficos
        )
        self.btn_graficos.grid(row=4, column=0, columnspan=2, pady=2)

        self.status_var = tk.StringVar(value='Parado')
        ttk.Label(frame_ctrl, textvariable=self.status_var, foreground='blue').grid(
            row=5, column=0, columnspan=2, **pad
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

    def _on_graficos(self):
        # uma só janela; se já existir, traz para a frente
        if getattr(self, 'janela_graficos', None) is not None:
            try:
                if self.janela_graficos.winfo_exists() and self.janela_graficos._ativa:
                    self.janela_graficos.lift()
                    return
            except tk.TclError:
                pass
        try:
            self.janela_graficos = JanelaGraficos(self, self.node)
        except ImportError:
            self.status_var.set('Falta o matplotlib: sudo apt install python3-matplotlib')

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