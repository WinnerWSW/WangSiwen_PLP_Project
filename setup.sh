#!/bin/bash

# 安装 Rust 编译器（用于需要 Rust 的 Python 库）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 更新 Rust 的环境变量
source $HOME/.cargo/env
