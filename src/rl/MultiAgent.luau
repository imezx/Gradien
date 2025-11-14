--!optimize 2
local Types = require("../Types")

local MultiAgent = {}
MultiAgent.__index = MultiAgent

function MultiAgent:getAgent(i: number): Types.Module
	assert(i >= 1 and i <= self._numAgents, ("MultiAgent: index %d out of range [1,%d]"):format(i, self._numAgents))
	return self._agents[i]
end

function MultiAgent:size(): number
	return self._numAgents
end

function MultiAgent:act(i: number, state: Types.Tensor, step: number): Types.Tensor
	local agent = self:getAgent(i)
	return agent:act(state, step)
end

function MultiAgent:actAll(states: { Types.Tensor }, step: number): { Types.Tensor }
	assert(#states == self._numAgents, "MultiAgent: actAll() expects one state per agent")
	local actions = table.create(self._numAgents)
	for idx = 1, self._numAgents do
		actions[idx] = self._agents[idx]:act(states[idx], step)
	end
	return actions
end

function MultiAgent:observe(i: number, t: Types.Tensor)
	local agent = self:getAgent(i)
	if agent.observe then
		agent:observe(t)
	end
end

function MultiAgent:observeAll(ts: { Types.Tensor })
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

function MultiAgent:trainStep(i: number): Types.Tensor?
	if i ~= nil then
		local agent = self:getAgent(i)
		if agent.trainStep then
			return agent:trainStep()
		else
			return nil
		end
	end
	return self:trainStepAll()
end

function MultiAgent:trainStepAll(): { Types.Tensor? }
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

function MultiAgent:parameters(): { Types.Tensor }
	local params = {}
	for idx = 1, self._numAgents do
		local agent = self._agents[idx]
		if agent.parameters then
			local ps = agent:parameters()
			if ps then
				for j = 1, #ps do
					table.insert(params, ps[j])
				end
			end
		end
	end
	return params
end

function MultiAgent:map(fn: (Types.Module, number) -> ())
	for idx = 1, self._numAgents do
		fn(self._agents[idx], idx)
	end
end

return function(cfg: { numAgents: number, makeAgent: (number) -> Types.Module }): typeof(setmetatable({}, MultiAgent))
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
	return setmetatable({
		_agents = agents,
		_numAgents = n,
	}, MultiAgent)
end
