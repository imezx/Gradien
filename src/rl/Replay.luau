--!optimize 2
local Tensor = require("../Tensor")

local function ensureBuffer(buf: { number }?, dim: number)
	if not buf or #buf ~= dim then
		return table.create(dim, 0)
	end
	return buf
end

local function copyStorage(src: { number }, dst: { number }, dim: number)
	for i = 1, dim do
		dst[i] = src[i]
	end
end

return function(capacity: number)
	local size, pos = 0, 1
	local S, A, R, NS, D = {}, {}, {}, {}, {}
	local stateDim: number? = nil
	local stateDType: string? = nil

	return {
		push = function(_, t)
			assert(t.state and t.nextState, "replay: state and nextState required")
			local srcState = t.state
			local srcNextState = t.nextState
			assert(srcState._storage and srcNextState._storage, "replay: states must be Tensors")
			if not stateDim then
				stateDim = #srcState._storage
				stateDType = srcState._dtype
			end
			local sd = stateDim
			local stateBuf = ensureBuffer(S[pos], sd)
			local nextStateBuf = ensureBuffer(NS[pos], sd)
			task.desynchronize()
			copyStorage(srcState._storage, stateBuf, sd)
			copyStorage(srcNextState._storage, nextStateBuf, sd)
			task.synchronize()
			S[pos] = stateBuf
			NS[pos] = nextStateBuf
			A[pos], R[pos], D[pos] = t.action, t.reward, t.done
			pos += 1
			if pos > capacity then
				pos = 1
			end
			if size < capacity then
				size += 1
			end
		end,
		canSample = function(_, n)
			return size >= n
		end,
		sample = function(_, n)
			assert(size >= n, "replay: not enough samples")
			assert(stateDim, "replay: buffer is empty")
			task.desynchronize()
			local sd = stateDim
			local dtype = stateDType
			local SX = Tensor.zeros({ sd, n }, dtype, false)
			local NSX = Tensor.zeros({ sd, n }, dtype, false)
			local AA, RR, DD = table.create(n, 1), table.create(n, 0), table.create(n, false)
			for j = 1, n do
				local i = math.random(1, size)
				AA[j], RR[j], DD[j] = A[i], R[i], D[i]
				local sArr = S[i]
				local nsArr = NS[i]
				for k = 1, sd do
					SX._storage[(k - 1) * n + j] = sArr[k]
					NSX._storage[(k - 1) * n + j] = nsArr[k]
				end
			end
			task.synchronize()
			return { S = SX, A = AA, R = RR, NS = NSX, D = DD }
		end,
	}
end
