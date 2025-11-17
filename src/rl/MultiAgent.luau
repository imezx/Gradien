--!optimize 2
local Types = require("../Types")
local Util = require("../Util")
local assert = Util.Assert

local function getAgent(self, i: number): Types.Module
	assert(i >= 1 and i <= self._numAgents, ("MultiAgent: index %d out of range [1,%d]"):format(i, self._numAgents))
	return self._agents[i]
end

local function size(self): number
	return self._numAgents
end

local function act(self, i: number, state: Types.Tensor, step: number): Types.Tensor
	return (getAgent(self, i) :: any):act(state, step)
end

local function actAll(self, states: { Types.Tensor }, step: number): { Types.Tensor }
	assert(#states == self._numAgents, "MultiAgent: actAll() expects one state per agent")
	local actions = table.create(self._numAgents)
	for idx = 1, self._numAgents do
		actions[idx] = self._agents[idx]:act(states[idx], step)
	end
	return actions
end

local function observe(self, i: number, t: Types.Tensor)
	local agent = getAgent(self, i)
	if agent.observe then
		agent:observe(t)
	end
end

local function observeAll(self, ts: { Types.Tensor })
	for idx = 1, self._numAgents do
		local t = ts[idx]
		if t then
			local agent = self._agents[idx]
			if agent.observe then
				agent:observe(t)
			end
		end
	end
end

local function trainStepAll(self): { Types.Tensor? }
	local out = table.create(self._numAgents)
	for idx = 1, self._numAgents do
		local agent = self._agents[idx]
		if agent.trainStep then
			out[idx] = agent:trainStep()
		else
			out[idx] = nil
		end
	end
	return out
end

local function trainStep(self, i: number?): Types.Tensor?
	if i ~= nil then
		local agent = getAgent(self, i)
		if agent.trainStep then
			return agent:trainStep()
		end
		return nil
	end
	return trainStepAll(self)
end

local function parameters(self): { Types.Tensor }
	local params = {}
	for idx = 1, self._numAgents do
		local agent = self._agents[idx]
		if agent.parameters then
			local ps = agent:parameters()
			if ps then
				for j = 1, #ps do
					params[#params + 1] = ps[j]
				end
			end
		end
	end
	return params
end

local function map(self, fn: (Types.Module, number) -> ())
	for idx = 1, self._numAgents do
		fn(self._agents[idx], idx)
	end
end

return function(cfg: { numAgents: number, makeAgent: (number) -> Types.Module })
	local n = assert(cfg.numAgents, "MultiAgent: cfg.numAgents required")
	assert(n >= 1, "MultiAgent: numAgents must be >= 1")
	local makeAgent = cfg.makeAgent
	assert(makeAgent, "MultiAgent: cfg.makeAgent required")
	local agents = table.create(n)
	for i = 1, n do
		local agent = makeAgent(i)
		assert(agent ~= nil, `MultiAgent: makeAgent({i}) returned nil`)
		agents[i] = agent
	end
	return {
		_agents = agents,
		_numAgents = n,
		getAgent = getAgent,
		size = size,
		act = act,
		actAll = actAll,
		observe = observe,
		observeAll = observeAll,
		trainStep = trainStep,
		trainStepAll = trainStepAll,
		parameters = parameters,
		map = map,
	}
end
