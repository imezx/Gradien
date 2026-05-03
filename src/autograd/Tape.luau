--!native
--!optimize 2
local Tensor = require("../Tensor")
local BLAS = require("../ops/BLAS")
local Types = require("../Types")
local Tape = {}
local gradEnabled = true

local function withTape(t: Types.Tensor, parents: { Types.Tensor }, back: (grad: Types.Tensor) -> ()): Types.Tensor
	if not gradEnabled then
		t._parents = nil
		t._backprop = nil
		t._requiresGrad = false
		return t
	end
	t._parents = parents
	t._backprop = back
	t._requiresGrad = false
	for _, p in parents do
		if p._requiresGrad then
			t._requiresGrad = true
			break
		end
	end
	return t
end

--@param out any @Tensor
--@param parents {any}
--@param back fun(grad: any)
function Tape.makeNode(out: Types.Tensor, parents: { Types.Tensor }, back: (grad: Types.Tensor) -> ()): Types.Tensor
	return withTape(out, parents, back)
end

-- y = ReLU(x)
--@param x @Tensor
function Tape.relu(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._requiresGrad)
	for i = 1, #x._storage do
		local v = x._storage[i]
		y._storage[i] = (v > 0) and v or 0
	end
	task.synchronize()
	return withTape(y, { x }, function(gy)
		if x._requiresGrad then
			task.desynchronize()
			if not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			if x._grad then
				for i = 1, #x._grad._storage do
					x._grad._storage[i] += (x._storage[i] > 0 and 1 or 0) * gy._storage[i]
				end
			end
			task.synchronize()
		end
	end)
end

-- y = A @ B
--@param x @Tensor
function Tape.matmul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	local Y = BLAS.matmul(A, B)
	return withTape(Y, { A, B }, function(gY: Types.Tensor)
		-- dA = gY @ B^T ; dB = A^T @ gY
		task.desynchronize()
		local M, K = A._shape[1], A._shape[2]
		local _, N = B._shape[1], B._shape[2]
		if A._requiresGrad then
			if not A._grad then
				A._grad = Tensor.zeros({ M, K })
			end
			local BT
			Tape.noGrad(function()
				BT = Tensor.transpose(B)
			end)
			local dA = BLAS.matmul(gY, BT)
			if A._grad then
				local dst = A._grad._storage
				local src = dA._storage
				for idx = 1, #dst do
					dst[idx] += src[idx]
				end
			end
		end
		if B._requiresGrad then
			if not B._grad then
				B._grad = Tensor.zeros({ K, N })
			end
			local AT
			Tape.noGrad(function()
				AT = Tensor.transpose(A)
			end)
			local dB = BLAS.matmul(AT, gY)
			if B._grad then
				local dst = B._grad._storage
				local src = dB._storage
				for idx = 1, #dst do
					dst[idx] += src[idx]
				end
			end
		end
		task.synchronize()
	end)
end

function Tape.backwardFrom(y: Types.Tensor, grad: Types.Tensor)
	task.desynchronize()
	local pending: { [Types.Tensor]: number } = {}
	local visitStack = { y }
	local seen = {}
	while #visitStack > 0 do
		local node = table.remove(visitStack)
		if not seen[node] then
			seen[node] = true
			if node._parents then
				for _, parent in node._parents do
					if parent._requiresGrad then
						pending[parent] = (pending[parent] or 0) + 1
						table.insert(visitStack, parent)
					end
				end
			end
		end
	end
	y._grad = grad
	local ready = { y }
	while #ready > 0 do
		local node = table.remove(ready)
		if node._requiresGrad then
			local g = node._grad
			if not g then
				g = Tensor.zeros(node._shape)
				node._grad = g
			end
			if node._backprop then
				node._backprop(g)
			end
			if node._parents then
				for _, parent in node._parents do
					if parent._requiresGrad then
						local remaining = pending[parent]
						if remaining then
							remaining -= 1
							if remaining == 0 then
								pending[parent] = nil
								if parent._backprop or parent._parents then
									table.insert(ready, parent)
								end
							else
								pending[parent] = remaining
							end
						else
							if parent._backprop or parent._parents then
								table.insert(ready, parent)
							end
						end
					end
				end
			end
		end
	end

	task.synchronize()
end

function Tape.noGrad(fn: () -> ()): ()
	gradEnabled = false
	fn()
	gradEnabled = true
end

function Tape.grad(f: (...any) -> Types.Tensor, inputs: { Types.Tensor }): { Types.Tensor? }
	local oldReq: { boolean } = {}
	for i, x in inputs do
		oldReq[i] = x._requiresGrad
		x._requiresGrad = true
		x._grad = nil
	end

	local y = f(table.unpack(inputs))
	if y == nil or y._shape == nil then
		error("Autograd.grad: f must return a Tensor")
	end

	Tape.backwardFrom(y, Tensor.ones(y._shape))

	local out: { Types.Tensor? } = {}
	for i, x in inputs do
		out[i] = x._grad
		x._requiresGrad = oldReq[i]
	end
	return out
end

return Tape
