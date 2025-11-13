--!optimize 2
local Tensor = require("../Tensor")

--@class PCA
--@field mean any -- {D,1}
--@field components any -- {D,K}
--@field k number
local PCA = {}
PCA.__index = PCA

local function matvec(D, C, vec) -- C * v
	local out = table.create(D, 0.0)
	for r = 1, D do
		local s = 0.0
		for c = 1, D do
			s += C._storage[(r - 1) * D + c] * vec[c]
		end
		out[r] = s
	end
	return out
end

local function normalize(D, v)
	local n2 = 0.0
	for i = 1, D do n2 += v[i] * v[i] end
	local inv = 1 / math.sqrt(math.max(n2, 1e-12))
	for i = 1, D do v[i] *= inv end
end

--@param X Tensor -- {D,N}
--@param K number -- components
--@param iters integer? -- default 50
--@return PCA
function PCA.new(X, K: number, iters: number?)
	local D, N = X._shape[1], X._shape[2]
	iters = iters or 50
	
	task.desynchronize()
	-- mean
	local mu = Tensor.zeros({ D, 1 }, X._dtype, false)
	for i = 1, D do
		local s = 0.0
		for j = 1, N do s += X._storage[(i - 1) * N + j] end
		mu._storage[i] = s / N
	end

	-- center: Xc = X - mu
	local Xc = Tensor.zeros({ D, N }, X._dtype, false)
	for i = 1, D do
		local m = mu._storage[i]
		for j = 1, N do
			Xc._storage[(i - 1) * N + j] = X._storage[(i - 1) * N + j] - m
		end
	end

	-- C = (1/N) Xc * Xc^T => {D,D}
	local C = Tensor.zeros({ D, D }, X._dtype, false)
	for i = 1, D do
		for k = i, D do
			local s = 0.0
			for j = 1, N do
				s += Xc._storage[(i - 1) * N + j] * Xc._storage[(k - 1) * N + j]
			end
			s /= N
			C._storage[(i - 1) * D + k] = s
			C._storage[(k - 1) * D + i] = s
		end
	end

	local comps = Tensor.zeros({ D, K }, X._dtype, false)

	for k = 1, K do
		local v = table.create(D, 0.0)
		for i = 1, D do v[i] = math.random() * 2 - 1 end
		normalize(D, v)

		for _ = 1, iters do
			local u = matvec(D, C, v)
			normalize(D, u)
			v = u
		end

		for i = 1, D do
			comps._storage[(i - 1) * K + k] = v[i]
		end

		local Cv = matvec(v)
		local num = 0.0
		for i = 1, D do num += v[i] * Cv[i] end
		local lambda = num
		for i = 1, D do
			for j = 1, D do
				C._storage[(i - 1) * D + j] -= lambda * v[i] * v[j]
			end
		end
	end
	task.synchronize()

	local self = setmetatable({}, PCA)
	self.mean = mu -- {D,1}
	self.components = comps -- {D,K}
	self.k = K
	return self
end

--@param X Tensor -- {D,B}
--@return Tensor -- {K,B}
function PCA:transform(X)
	task.desynchronize()
	local D, B = X._shape[1], X._shape[2]
	local K = self.k
	local Y = Tensor.zeros({ K, B }, X._dtype, X._requiresGrad)
	for j = 1, B do
		for k = 1, K do
			local s = 0.0
			for i = 1, D do
				local xij = X._storage[(i - 1) * B + j]
				local mu_i = self.mean._storage[i]
				local w = self.components._storage[(i - 1) * K + k]
				s += (xij - mu_i) * w
			end
			Y._storage[(k - 1) * B + j] = s
		end
	end
	task.synchronize()
	return Y
end

return PCA.new
