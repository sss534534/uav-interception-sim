#!/bin/bash
#
# PX4 SITL 环境安装脚本
# PX4 SITL Environment Setup Script
#
# 在 Ubuntu 上安装 PX4 SITL + Gazebo + MAVSDK 的完整环境

set -e

echo "=========================================="
echo "PX4 SITL Environment Setup"
echo "=========================================="

# 检查 Ubuntu 版本
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "[WARNING] This script is designed for Ubuntu"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "[1/6] Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    python3-pip \
    python3-venv \
    ninja-build \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libeigen3-dev \
    libxml2-utils \
    python3-jinja2 \
    python3-numpy \
    python3-toml \
    python3-yaml \
    python3-argcomplete \
    python3-serial \
    python3-empy \
    python3-pkgconfig \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgazebo-dev \
    gazebo11 \
    libgazebo11 \
    protobuf-compiler \
    libprotobuf-dev \
    libprotoc-dev \
    ros-noetic-mavros \
    ros-noetic-mavros-extras

echo ""
echo "[2/6] Installing Python packages..."
pip3 install --user mavsdk asyncio numpy matplotlib

echo ""
echo "[3/6] Cloning PX4 Autopilot..."
PX4_DIR="$HOME/PX4-Autopilot"
if [ -d "$PX4_DIR" ]; then
    echo "PX4 directory already exists. Updating..."
    cd "$PX4_DIR"
    git pull
else
    git clone https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR" --recursive
fi

echo ""
echo "[4/6] Building PX4 SITL..."
cd "$PX4_DIR"
make px4_sitl gazebo

echo ""
echo "[5/6] Setting up environment..."
# 添加到 .bashrc
if ! grep -q "PX4-Autopilot" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# PX4 SITL" >> ~/.bashrc
    echo "export PX4_DIR=$HOME/PX4-Autopilot" >> ~/.bashrc
    echo "source \$PX4_DIR/Tools/simulation/gazebo/setup_gazebo.bash \$PX4_DIR \$PX4_DIR/build/px4_sitl_default" >> ~/.bashrc
    echo "export ROS_PACKAGE_PATH=\$ROS_PACKAGE_PATH:\$PX4_DIR:\$PX4_DIR/Tools/simulation/gazebo/sitl_gazebo" >> ~/.bashrc
    echo "export GAZEBO_PLUGIN_PATH=\$GAZEBO_PLUGIN_PATH:\$PX4_DIR/build/px4_sitl_default/build_gazebo" >> ~/.bashrc
    echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:\$PX4_DIR/Tools/simulation/gazebo/sitl_gazebo/models" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$PX4_DIR/build/px4_sitl_default/build_gazebo" >> ~/.bashrc
fi

echo ""
echo "[6/6] Creating run scripts..."
mkdir -p "$HOME/px4_interception_scripts"

# 创建启动脚本
cat > "$HOME/px4_interception_scripts/start_px4_sitl.sh" << 'EOF'
#!/bin/bash
# 启动 PX4 SITL 双机仿真

PX4_DIR="$HOME/PX4-Autopilot"

# 启动第一架无人机 (UDP:14540)
HEADLESS=1 $PX4_DIR/build/px4_sitl_default/bin/px4 -d \
    $PX4_DIR/build/px4_sitl_default/etc \
    -s $PX4_DIR/build/px4_sitl_default/etc/init.d-posix/rcS \
    -i 0 &

# 启动第二架无人机 (UDP:14541)
HEADLESS=1 $PX4_DIR/build/px4_sitl_default/bin/px4 -d \
    $PX4_DIR/build/px4_sitl_default/etc \
    -s $PX4_DIR/build/px4_sitl_default/etc/init.d-posix/rcS \
    -i 1 &

echo "PX4 SITL started on ports 14540 and 14541"
echo "Run 'ps aux | grep px4' to see processes"
echo "Run 'killall px4' to stop"
EOF

chmod +x "$HOME/px4_interception_scripts/start_px4_sitl.sh"

# 创建 MAVSDK 服务器启动脚本
cat > "$HOME/px4_interception_scripts/start_mavsdk_server.sh" << 'EOF'
#!/bin/bash
# 启动 MAVSDK 服务器

# 无人机 1 (端口 50051 -> 14540)
mavsdk_server -p 50051 udp://:14540 &

# 无人机 2 (端口 50052 -> 14541)
mavsdk_server -p 50052 udp://:14541 &

echo "MAVSDK servers started on ports 50051 and 50052"
EOF

chmod +x "$HOME/px4_interception_scripts/start_mavsdk_server.sh"

# 创建一键启动脚本
cat > "$HOME/px4_interception_scripts/start_simulation.sh" << 'EOF'
#!/bin/bash
# 一键启动完整仿真环境

echo "Starting PX4 SITL + MAVSDK..."

# 停止已有进程
killall px4 2>/dev/null || true
killall mavsdk_server 2>/dev/null || true
sleep 2

# 启动 PX4
$HOME/px4_interception_scripts/start_px4_sitl.sh
sleep 5

# 启动 MAVSDK 服务器
$HOME/px4_interception_scripts/start_mavsdk_server.sh
sleep 2

echo ""
echo "Simulation environment ready!"
echo "Now run: python3 run_px4_interception.py"
EOF

chmod +x "$HOME/px4_interception_scripts/start_simulation.sh"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Source your .bashrc: source ~/.bashrc"
echo "2. Start simulation: $HOME/px4_interception_scripts/start_simulation.sh"
echo "3. Run interception test: python3 run_px4_interception.py"
echo ""
echo "Or run in simulation mode (no PX4 required):"
echo "  python3 run_px4_interception.py"
echo ""
