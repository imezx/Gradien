import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Gradien",
  description: "Parallelized Machine Learning for Roblox.",
  base: '/Gradien/',
  lang: 'en-US',
  cleanUrls: true,
  themeConfig: {
    search: {
      provider: 'local'
    },
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api/core/tensor' }
    ],
    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Core Concepts', link: '/guide/core-concepts' },
          { text: 'The Trainer Loop', link: '/guide/trainer' },
          { text: 'Saving & Loading', link: '/guide/state' },
          { text: 'Creating a Neural Network', link: '/guide/create-nn' },
        ]
      },
      {
        text: 'Core',
        collapsed: false,
        items: [
          { text: 'Tensor', link: '/api/core/tensor' },
          { text: 'Autograd (Tape)', link: '/api/core/autograd' },
          { text: 'Math Operations', link: '/api/core/math' }
        ]
      },
      {
        text: 'Neural Network',
        collapsed: false,
        items: [
          { text: 'Layers', link: '/api/nn/layers' },
          { text: 'Activations', link: '/api/nn/activations' },
          { text: 'Loss Functions', link: '/api/nn/losses' },
          { text: 'Initializers', link: '/api/nn/initializers' }
        ]
      },
      {
        text: 'Optimization',
        collapsed: false,
        items: [
          { text: 'Optimizers', link: '/api/optim/optimizers' },
          { text: 'Wrappers (EMA, Lookahead)', link: '/api/optim/wrappers' },
          { text: 'LR Schedulers', link: '/api/optim/schedulers' },
          { text: 'Gradient Clipping', link: '/api/optim/gradclip' }
        ]
      },
      {
        text: 'Experimental',
        collapsed: false,
        items: [
          { text: 'Quantum-Inspired (QIMHNN)', link: '/api/experimental/qimhnn' },
          { text: 'Swarm Optimizers', link: '/api/experimental/optimizers' },
          { text: 'Neural Networks (KAN)', link: '/api/experimental/nn' },
          { text: 'Mamba Block', link: '/api/experimental/mamba' },
          { text: 'Feudal', link: '/api/experimental/feudal' }
        ]
      },
      {
        text: 'Reinforcement Learning',
        collapsed: false,
        items: [
          { text: 'RL Agents', link: '/api/rl/agents' }
        ]
      },
      {
        text: 'Utilities',
        collapsed: true,
        items: [
          { text: 'Data Pipeline', link: '/api/utils/data' },
          { text: 'Model Tools', link: '/api/utils/model-tools' },
          { text: 'Visualization', link: '/api/utils/visualization' },
          { text: 'Metrics & Debugging', link: '/api/utils/debug' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/imezx/Gradien' },
      { icon: 'discord', link: 'https://discord.gg/qnSfEZ6bZK' },
    ]
  }
})