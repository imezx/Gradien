# Generative Reasoning (GRAM) <Badge type="warning" text="Experimental" />

**Generative Recursive Reasoning Model**

`GRAM` is an implementation of the **Generative Recursive Reasoning Model** (arXiv:2605.19376). It models the agent's internal reasoning process as a stochastic latent trajectory, scaling inference-time computation dynamically along both **depth** (adaptive recursive steps) and **width** (parallel trajectory sampling).

By structuring computation around a low-level "System 1" loop and a high-level "System 2" planner, `GRAM` allows models to spend more compute time on harder tokens or decision states, guided by an auxiliary Latent Process Reward Model (LPRM).

---

## Architectural Blueprint

```mermaid
graph TD
    X[Input Tensor x] --> Context[h_prev + x]
    Context --> S1[System 1 Inner Loop: f_L]
    S1 -- Iterates K times --> S1_End[Refined Latent l_t]
    S1_End --> S2[System 2 Outer Loop: f_H]
    H_Prev[h_prev] --> S2
    S2 --> Proposal[Proposal State u_t]
    
    Proposal --> Prior[Prior Net: p]
    Prior --> Prior_Dist[p = N(mu_p, sigma_p)]
    
    subgraph Posterior Mode (Training Only)
        Y[Target y] --> Concat[concat(u_t, y)]
        Concat --> Post[Posterior Net: q]
        Post --> Post_Dist[q = N(mu_q, sigma_q)]
    end
    
    Prior_Dist -.-> Sample[Reparameterized Sample]
    Post_Dist -.-> Sample
    
    Sample --> H_t[Next Latent h_t]
    H_t --> ACT[ACT Halting Head]
    H_t --> LPRM[LPRM Score Head]
    H_t --> LM[LM Logits Head]
    
    ACT -- Halt Prob > 0.5 --> Exit[Exit Loop early]
```

### Key Components

1. **System 1 (Inner Loop, $K$ steps)**: Low-level state refinement $l_{t,k} = f_L(l_{t,k-1} + h_{t-1} + e_x)$. Represents fast, intuitive processing.
2. **System 2 (Outer Loop, $T$ steps)**: High-level planning $u_t = f_H(h_{t-1} + l_t)$. Computes proposals for the next reasoning step.
3. **Stochastic Transition**: Adds reparameterized noise $\epsilon_t \sim \mathcal{N}(\mu, \sigma^2 I)$ to the proposal state ($h_t = u_t + \epsilon_t$). This allows the model to explore reasoning pathways while remaining fully differentiable.
4. **Adaptive Computation Time (ACT)**: Halts the outer loop dynamically during prior mode inference when the halting probability exceeds $0.5$.
5. **Latent Process Reward Model (LPRM)**: Predicts the expected correctness of the current reasoning state, enabling best-of-$N$ path selection during search.

---

## Constructor

Located in `Gradien.Experimental.Models.GRAM`.

::: code-group
```lua [Definition]
(config: GRAMConfig) -> GRAMModule
```

```lua [Configuration]
type GRAMConfig = {
    embedDim: number,       -- Embedding / state dimension (required)
    vocabSize: number,      -- Output vocabulary / action dimension (required)
    numHeads: number?,      -- Number of attention heads in f_L/f_H blocks (default: 8)
    mlpDim: number?,        -- Hidden dimension for GatedMLP layers (default: embedDim)
    K: number?,             -- Number of inner loop (System 1) refinement steps (default: 4)
    T: number?,             -- Max number of outer loop (System 2) planning steps (default: 3)
    beta: number?,          -- KL divergence penalty coefficient (default: 0.05)
    alpha: number?,         -- KL balancing weight for prior/posterior (default: 0.8)
    useAttention: boolean?, -- Enable Multi-Head Attention blocks in f_L/f_H (default: true)
}
```

```lua [Instantiation]
local Gradien = require(game.ReplicatedStorage.Gradien)

local model = Gradien.Experimental.Models.GRAM({
    embedDim = 64,
    vocabSize = 6,
    numHeads = 4,
    K = 2,
    T = 2,
    useAttention = false -- Set to false to avoid self-attention overhead on shape [D, 1] states
})
```
:::

---

## Methods

### `forward`
Executes the recursive latent transition loop. 

```lua
model:forward(x: Tensor, y: Tensor?) -> Tensor
```

* **Parameters**:
  * `x` (`Tensor`): Input token/state representation of shape `[embedDim, L]`.
  * `y` (`Tensor`?, optional): Ground-truth target sequence of shape `[vocabSize, L]`.
* **Returns**: Logits `Tensor` of shape `[vocabSize, L]`.

> [!NOTE]
> **Prior vs. Posterior Mode**
> - **Posterior Mode (Training)**: Triggered when `y` is provided and `model._train` is `true`. `GRAM` samples latents from the posterior $q(\cdot | u_t, y)$ and computes the Variational KL divergence against the prior $p(\cdot | u_t)$, adding a balanced KL loss to the graph.
> - **Prior Mode (Inference)**: Triggered when `y` is omitted or `model._train` is `false`. `GRAM` samples latents from the prior $p(\cdot | u_t)$ and runs Adaptive Computation Time (ACT) to halt reasoning early if it reaches confidence.

---

### `parameters`
Returns a list of all learnable parameter Tensors.

```lua
model:parameters() -> {Tensor}
```

* **Returns**: Array of trainable parameter `Tensor` objects.

---

### `train`
Sets the module's training mode state.

```lua
model:train(mode: boolean?)
```

* **Parameters**:
  * `mode` (`boolean`?, optional): `true` to enable training mode, `false` to disable. Defaults to `true` if omitted.

---

### `generate`
Performs parallel width-scaling (best-of-$N$ search) over multiple stochastic reasoning paths.

```lua
model:generate(x: Tensor, N: number) -> (Tensor, number)
```

* **Parameters**:
  * `x` (`Tensor`): Input state representation of shape `[embedDim, L]`.
  * `N` (`number`): The number of independent trajectory reasoning paths to sample.
* **Returns**:
  1. `best_logits` (`Tensor`): The output action logits from the reasoning path with the highest predicted reward.
  2. `best_score` (`number`): The highest LPRM value scored among all sampled paths.

---

## Mathematical Formulation

### 1. State Refinement Loop
At each outer step $t \in \{1, \dots, T\}$, `GRAM` initializes the inner loop with $l_{t,0} = l_{t-1}$. For $k = 1, \dots, K$, the low-level block $f_L$ processes the context:
$$l_{t,k} = f_L(l_{t,k-1} + h_{t-1} + e_x)$$
The final inner state $l_t = l_{t,K}$ is passed to the high-level block $f_H$ to generate the proposal:
$$u_t = f_H(h_{t-1} + l_t)$$

### 2. Stochastic Latents & KL Balancing
The model projects $u_t$ to prior parameters:
$$p(h_t | h_{t-1}, x) = \mathcal{N}(\mu_p(u_t), \sigma_p^2(u_t) I)$$
During training, the posterior network integrates the target label $y$:
$$q(h_t | h_{t-1}, x, y) = \mathcal{N}(\mu_q(u_t, y), \sigma_q^2(u_t, y) I)$$
To optimize the variational bounds, `GRAM` computes the KL divergence using **KL Balancing** with weight $\alpha$:
$$\mathcal{L}_{\text{KL}} = \beta \left( \alpha \cdot D_{\text{KL}}(q \parallel \text{detach}(p)) + (1 - \alpha) \cdot D_{\text{KL}}(\text{detach}(q) \parallel p) \right)$$
This prevents the prior from collapsing to the posterior too quickly, maintaining diverse exploration.

### 3. Adaptive Halting (ACT)
During inference, the model projects the latent state $h_t$ to a halting logit. The halting probability at step $t$ is:
$$a_t = \sigma(\text{mean}(\text{act\_head}(h_t)))$$
The computation stops early at step $t$ if $a_t > 0.5$.

---

## Examples

### Complete Training and Generation Workflow

```lua
local Gradien = require(game.ReplicatedStorage.Gradien)
local Optim = Gradien.Optim
local Tensor = Gradien.Tensor

-- 1. Create Model
local model = Gradien.Experimental.Models.GRAM({
    embedDim = 32,
    vocabSize = 4,
    K = 2,
    T = 2,
    useAttention = false
})

local optimizer = Optim.AdamW(model:parameters(), 0.001)

-- 2. Training Step (Posterior Tracking)
model:train(true)

local inputState = Tensor.randn({32, 1}, true) -- [embedDim, L]
local targetAction = Tensor.zeros({4, 1}, false) -- One-hot target [vocabSize, L]
targetAction._storage[2] = 1.0 -- Target action index 2

-- Forward pass in posterior mode (passing target targetAction)
local logits = model:forward(inputState, targetAction)

-- Compute loss (e.g., Mean Squared Error or CrossEntropy)
local diff = Tensor.zeros(logits._shape, true)
for i = 1, #logits._storage do
    diff._storage[i] = logits._storage[i] - targetAction._storage[i]
end
local loss = Tensor.fromArray({0}, {1}, true)
for i = 1, #diff._storage do
    loss._storage[1] = loss._storage[1] + diff._storage[i] * diff._storage[i]
end

-- Backward pass & Optimize
loss:backward()
optimizer:step()
optimizer:zeroGrad()

-- 3. Inference and Width-Scaling (Best-of-N Trajectories)
model:train(false)

local evalState = Tensor.randn({32, 1}, false)
local bestLogits, bestReward = model:generate(evalState, 8)

print("Best Trajectory LPRM Score:", bestReward)
```
