--!optimize 2
local Tensor = require("../Tensor")
local Buffer = require("../Util/Buffer")
local Util = require("../Util")
local assert = Util.Assert

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
			S[pos] = Buffer.writeDenseArray(srcState._storage, stateDType, S[pos])
			NS[pos] = Buffer.writeDenseArray(srcNextState._storage, stateDType, NS[pos])
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
				Buffer.copyDenseToColumn(S[i], dtype, SX._storage, j, n, sd)
				Buffer.copyDenseToColumn(NS[i], dtype, NSX._storage, j, n, sd)
			end
			task.synchronize()
			return { S = SX, A = AA, R = RR, NS = NSX, D = DD }
		end,
	}
end
