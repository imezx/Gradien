---
layout: home

hero:
  name: "Gradien"
  text: "Parallelized Machine Learning for Roblox"
  tagline: "A strictly-typed, high-performance ML & DL library powered by Parallel Luau."
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/core/tensor
    - theme: alt
      text: View on GitHub
      link: https://github.com/EternityDevs/Gradien

features:
  - icon: 🚀
    title: Parallel-first Compute
    details: Heavy numeric operations (matrix multiplication, element-wise math) are dispatched to parallel threads, unlocking performance impossible in serial Luau.
  - icon: 🛡️
    title: Strictly Typed
    details: Built with robust Luau type checking. Tensors, Modules, and Optimizers have strict shapes and definitions for excellent autocomplete and safety.
  - icon: 🧠
    title: Deep Reinforcement Learning
    details: Production-ready implementations of PPO, A2C, DoubleDQN, and DQL with a unified Agent API.
  - icon: 👁️
    title: Real-Time Visualization
    details: Debug your networks visually in 3D world-space or 2D UI with live weight and activation heatmaps.
---