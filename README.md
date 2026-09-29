# RobotFactory4.0

Simulação em Gazebo (gz-sim) + ROS 2 do robô diferencial da Robot@Factory 4.0:
2 motores N20 (12 V, 200 rpm, 1:150, encoder Hall AB 1050 PPR/canal), 2 rodas-guia
e IMU BNO055.

## Convenções
- `base_link`: **+X frente, +Y esquerda, +Z cima** (REP-103). O IMU (`imu_link`) está
  alinhado com `base_link`; o mundo é ENU (X = Este) e o robô nasce virado para +X.
- Rodas motrizes em ±Y com eixo Y; rotação positiva das duas => avança em +X.
- O chassis (STL) está rodado 90° em yaw no xacro (`chassis_yaw`); é simétrico.

## Dependências
`ros_gz_sim`, `ros_gz_bridge`, `robot_state_publisher`, `xacro`, `imu_filter_madgwick`;
para o script: `python3-tk`, `python3-matplotlib`, `python3-numpy`.

## Compilar e correr
```bash
colcon build --packages-select raf4_description && source install/setup.bash
ros2 launch raf4_description gazebo.launch.py          # terminal 1
python3 scripts/mover_e_ler_imu.py                      # terminal 2 (GUI)
```

## Parâmetros a ajustar (em `raf4.urdf.xacro`)
- `base_mass`, `wheel_mass`: **placeholders**, pesar o robô/roda reais.
- `wheel_visual_shift`: 6.5 mm (a malha da roda tem a origem na face). Pôr 0 se as
  posições das rodas medidas no CAD forem a face e não o centro.
- Limites do `DiffDrive` (0.40 m/s, acelerações): estimativas a partir do N20.
- Bias do acelerómetro (0.1 m/s²): suposição, o datasheet só dá o valor em bruto (±80 mg).
