--!optimize 2
local Tensor = require("../Tensor")
local Types = require("../Types")

local profiler = {
	enabled = false,
	total = 0,
	calls = 0,
	last = 0,
	max = 0,
}

local function recordProfile(startTime: number?)
	if not profiler.enabled or not startTime then
		return
	end
	local dt = os.clock() - startTime
	profiler.total += dt
	profiler.calls += 1
	profiler.last = dt
	if dt > profiler.max then
		profiler.max = dt
	end
end

local function matmul(A: Types.Tensor, B: Types.Tensor): Types.Tensor
	local profStart = profiler.enabled and os.clock() or nil
	local M, K = A._shape[1], A._shape[2]
	local K2, N = B._shape[1], B._shape[2]
	assert(K == K2, "matmul: inner dims mismatch")
	local C = Tensor.zeros({ M, N }, A._dtype, A._requiresGrad or B._requiresGrad)
	task.desynchronize()
	local as, bs, cs = A._storage, B._storage, C._storage
	for i = 1, M do
		local aRowOffset = (i - 1) * K
		local cRowOffset = (i - 1) * N
		for k = 1, K do
			local aVal = as[aRowOffset + k]
			if aVal ~= 0 then
				local bRowOffset = (k - 1) * N
				local baseC = cRowOffset
				local limit = N - (N % 4)
				local j = 1
				while j <= limit do
					local idxC = baseC + j
					local idxB = bRowOffset + j
					cs[idxC] += aVal * bs[idxB]
					cs[idxC + 1] += aVal * bs[idxB + 1]
					cs[idxC + 2] += aVal * bs[idxB + 2]
					cs[idxC + 3] += aVal * bs[idxB + 3]
					j += 4
				end
				for jj = limit + 1, N do
					cs[baseC + jj] += aVal * bs[bRowOffset + jj]
				end
			end
		end
	end
	task.synchronize()
	recordProfile(profStart)
	return C
end

local function enableProfiling(flag: boolean)
	profiler.enabled = flag
end

local function resetProfiling()
	profiler.total = 0
	profiler.calls = 0
	profiler.last = 0
	profiler.max = 0
end

local function getProfile()
	local avg = profiler.calls > 0 and profiler.total / profiler.calls or 0
	return {
		enabled = profiler.enabled,
		calls = profiler.calls,
		totalSeconds = profiler.total,
		avgSeconds = avg,
		lastSeconds = profiler.last,
		maxSeconds = profiler.max,
	}
end

return {
	matmul = matmul,
	enableProfiling = enableProfiling,
	resetProfiling = resetProfiling,
	getProfile = getProfile,
}
