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

return function(capacity: number, alpha: number, beta0: number)
	alpha = alpha or 0.6
	beta0 = beta0 or 0.4
	local size, pos = 0, 1
	local S, A, R, NS, D = {}, {}, {}, {}, {}
	local N = 1
	while N < capacity do
		N *= 2
	end
	local tree = table.create(2 * N, 0)
	local maxPrio = 1
	local stateDim: number? = nil
	local stateDType: string? = nil

	local function updateLeaf(i, p)
		task.desynchronize()
		local idx = N + i - 1
		tree[idx] = p
		idx //= 2
		while idx >= 1 do
			tree[idx] = tree[idx * 2] + tree[idx * 2 + 1]
			idx //= 2
		end
		task.synchronize()
	end
	local function sampleOne(u)
		task.desynchronize()
		local idx = 1
		while idx < N do
			local left = idx * 2
			if u <= tree[left] then
				idx = left
			else
				u -= tree[left]
				idx = left + 1
			end
		end
		task.synchronize()
		return idx - N + 1
	end
	return {
		push = function(_, t)
			assert(t.state and t.nextState, "prio replay: state and nextState required")
			if not stateDim then
				stateDim = #t.state._storage
				stateDType = t.state._dtype
			end
			local sd = stateDim
			local stateBuf = ensureBuffer(S[pos], sd)
			local nextStateBuf = ensureBuffer(NS[pos], sd)
			task.desynchronize()
			copyStorage(t.state._storage, stateBuf, sd)
			copyStorage(t.nextState._storage, nextStateBuf, sd)
			task.synchronize()
			S[pos] = stateBuf
			NS[pos] = nextStateBuf
			A[pos], R[pos], D[pos] = t.action, t.reward, t.done
			updateLeaf(pos, maxPrio ^ alpha)
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
			assert(size >= n, "prio replay: not enough samples")
			assert(stateDim, "prio replay: buffer is empty")
			task.desynchronize()
			local sd = stateDim
			local dtype = stateDType
			local SX = Tensor.zeros({ sd, n }, dtype, false)
			local NSX = Tensor.zeros({ sd, n }, dtype, false)
			local AA, RR, DD = table.create(n, 1), table.create(n, 0), table.create(n, false)
			local idxs, w = table.create(n, 1), table.create(n, 0.0)
			local tot = tree[1]
			local segment = tot / n
			for j = 1, n do
				local u = (j - 1) * segment + math.random() * segment
				local i = sampleOne(u)
				if i > size then
					i = size
				end
				idxs[j] = i
				local p = tree[N + i - 1]
				local P = p / tot
				w[j] = ((size * P) ^ -beta0)
				AA[j], RR[j], DD[j] = A[i], R[i], D[i]
				local sArr = S[i]
				local nsArr = NS[i]
				for k = 1, sd do
					SX._storage[(k - 1) * n + j] = sArr[k]
					NSX._storage[(k - 1) * n + j] = nsArr[k]
				end
			end
			local maxw = 0
			for j = 1, n do
				if w[j] > maxw then
					maxw = w[j]
				end
			end
			for j = 1, n do
				w[j] /= (maxw > 0 and maxw or 1)
			end
			task.synchronize()
			return { indices = idxs, weights = w, S = SX, A = AA, R = RR, NS = NSX, D = DD }
		end,
		updatePriorities = function(_, idxs, prios)
			for k = 1, #idxs do
				local i = idxs[k]
				local p = prios[k]
				if p > maxPrio then
					maxPrio = p
				end
				updateLeaf(i, p ^ alpha)
			end
		end,
	}
end
