--[[
Gradien (by @Eternity_Devs)
A strictly-typed, optimized parallel-first ML & DL library for Roblox.
Version 1.4.0-rc4
]]
--!strict
local Gradien = {}

Gradien.Autograd = require("@self/autograd/Tape")
Gradien.Init = require("@self/Initializer")
Gradien.Metrics = require("@self/Metrics")
Gradien.GradClip = require("@self/GradClip")
Gradien.Trainer = require("@self/Trainer")
Gradien.State = require("@self/State")
Gradien.Types = require("@self/Types")
Gradien.Util = require("@self/Util")
Gradien.Tensor = require("@self/Tensor")
Gradien.Debug = require("@self/Debug")

Gradien.Experimental = {
	Models = {
		QIMHNN = require("@self/experimental/models/QIMHNN"),
	},
	Optim = {
		Metaheuristic = require("@self/experimental/optim/Metaheuristic"),
		SwarmPSO = require("@self/experimental/optim/SwarmPSO"),
	},
}

Gradien.Models = {
	MLP = require("@self/models/MLP"),
	ResMLP = require("@self/models/ResMLP"),
	ConvNet = require("@self/models/ConvNet"),
	TransformerEncoder = require("@self/models/TransformerEncoder"),
	SequenceClassifier = require("@self/models/SequenceClassifier"),
	AutoEncoder = require("@self/models/AutoEncoder"),
	Utils = require("@self/models/Utils"),
}
Gradien.Ops = {
	Math = require("@self/ops/Math"),
	BLAS = require("@self/ops/BLAS"),
	Reduce = require("@self/ops/Reduce"),
	Conv2d = require("@self/ops/Conv2d"),
	Pool = require("@self/ops/Pool"),
	Softmax = require("@self/ops/Softmax"),
	AvgPool2d = require("@self/ops/AvgPool2d"),
	Scan = require("@self/ops/Scan"),
}
Gradien.NN = {
	Linear = require("@self/nn/Linear"),
	NoisyLinear = require("@self/nn/NoisyLinear"),
	Activations = require("@self/nn/Activations"),
	LayerNorm = require("@self/nn/LayerNorm"),
	BatchNorm1d = require("@self/nn/BatchNorm1d"),
	BatchNorm2d = require("@self/nn/BatchNorm2d"),
	Embedding = require("@self/nn/Embedding"),
	Attention = require("@self/nn/Attention"),
	RMSNorm = require("@self/nn/RMSNorm"),
	RNN = require("@self/nn/RNN"),
	GRU = require("@self/nn/GRU"),
	LSTM = require("@self/nn/LSTM"),
	MambaBlock = require("@self/nn/Mamba"),
	Sequential = require("@self/nn/Sequential"),
	Dropout = require("@self/nn/Dropout"),
	Fused = require("@self/nn/Fused"),
	Softmax = require("@self/nn/Softmax"),
	Losses = require("@self/nn/Losses"),
	Functional = require("@self/nn/Functional"),
	GatedMLP = require("@self/nn/GatedMLP"),
	DropPath = require("@self/nn/DropPath"),
	Residual = require("@self/nn/Residual"),
	SwiGLU = require("@self/nn/SwiGLU"),
	Conv2d = require("@self/nn/Conv2d"),
	MaxPool2d = require("@self/nn/MaxPool2d"),
	AvgPool2d = require("@self/nn/AvgPool2d"),
	Flatten = require("@self/nn/Flatten"),
}
Gradien.Optim = {
	SGD = require("@self/optim/SGD"),
	Adam = require("@self/optim/Adam"),
	AdamW = require("@self/optim/AdamW"),
	EMA = require("@self/optim/EMA"),
	RMSProp = require("@self/optim/RMSProp"),
	Adagrad = require("@self/optim/Adagrad"),
	Adafactor = require("@self/optim/Adafactor"),
	Accumulated = require("@self/optim/Accumulated"),
	Schedulers = require("@self/optim/Schedulers"),
	Lion = require("@self/optim/Lion"),
	Lookahead = require("@self/optim/Lookahead"),
	Sophia = require("@self/optim/Sophia"),
	Muon = require("@self/optim/Muon"),
}
Gradien.RL = {
	Replay = require("@self/rl/Replay"),
	UniformReplay = require("@self/rl/UniformReplay"),
	PrioritizedReplay = require("@self/rl/PrioritizedReplay"),
	DQL = require("@self/rl/DQL"),
	DoubleDQN = require("@self/rl/DoubleDQN"),
	C51DQN = require("@self/rl/C51DQN"),
	BDQ = require("@self/rl/BDQ"),
	Dueling = require("@self/rl/Dueling"),
	NStep = require("@self/rl/NStep"),
	LambdaBuffer = require("@self/rl/LambdaBuffer"),
	A2C = require("@self/rl/A2C"),
	PPO = require("@self/rl/PPO"),
	MultiAgent = require("@self/rl/MultiAgent"),
}
Gradien.Classic = {
	SoftmaxRegression = require("@self/classic/SoftmaxRegression"),
	LinearRegression = require("@self/classic/LinearRegression"),
	Ridge = require("@self/classic/Ridge"),
	LogisticRegression = require("@self/classic/LogisticRegression"),
	KMeans = require("@self/classic/KMeans"),
	KNN = require("@self/classic/KNN"),
	GaussianNB = require("@self/classic/GaussianNB"),
	MultinomialNB = require("@self/classic/MultinomialNB"),
}
Gradien.Preprocess = {
	StandardScaler = require("@self/preprocess/StandardScaler"),
	MinMaxScaler = require("@self/preprocess/MinMaxScaler"),
	OneHot = require("@self/preprocess/OneHot"),
	PCA = require("@self/preprocess/PCA"),
	SinusoidalPE = require("@self/preprocess/SinusoidalPE"),
}
Gradien.Data = {
	TensorDataset = require("@self/data/TensorDataset"),
	DataLoader = require("@self/data/DataLoader"),
	Split = require("@self/data/Split"),
	KFold = require("@self/data/KFold"),
}
Gradien.Extra = {
	RND = require("@self/extra/RND"),
}
Gradien.Tools = {
	Gradcheck = require("@self/tools/Gradcheck"),
}

return Gradien
