#!/bin/bash

# 安装 Rust 编译器
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 将 Rust 添加到 PATH
source $HOME/.cargo/env

# 确保 pip 最新
pip install --upgrade pip
