--!optimize 2
local Tensor = require("../Tensor")

return function(capacity: number)
	local size, pos = 0, 1
	local S, A, R, NS, D = {}, {}, {}, {}, {}
	return {
		push = function(_, t)
			S[pos], A[pos], R[pos], NS[pos], D[pos] = t.state, t.action, t.reward, t.nextState, t.done
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
			task.desynchronize()
			local sd = S[1]._shape[1]
			local SX = Tensor.zeros({ sd, n })
			local NSX = Tensor.zeros({ sd, n })
			local AA, RR, DD = table.create(n, 1), table.create(n, 0), table.create(n, false)
			for j = 1, n do
				local i = math.random(1, size)
				AA[j], RR[j], DD[j] = A[i], R[i], D[i]
				for k = 1, sd do
					SX._storage[(k - 1) * n + j] = S[i]._storage[k]
					NSX._storage[(k - 1) * n + j] = NS[i]._storage[k]
				end
			end
			task.synchronize()
			return { S = SX, A = AA, R = RR, NS = NSX, D = DD }
		end,
	}
end
