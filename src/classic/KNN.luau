--!optimize 2
local Types = require("../Types")

--@class KNN
--@field X any
--@field y {number}
--@field k integer
local KNN = {}
KNN.__index = KNN

--@param X Tensor -- {D,N} training
--@param y {number} -- length N, class indices (1..C)
--@param k integer? -- neighbors (default 5)
--@return KNN
function KNN.new(X: Types.Tensor, y: { number }, k: number?)
	assert(#y == X._shape[2], "KNN: y length must equal N")
	local self = setmetatable({}, KNN)
	self.X, self.y, self.k = X, y, k or 5
	return self
end

--@param Q Tensor -- {D,B}
--@return {number} -- {B} predicted classes
function KNN:predict(Q: Types.Tensor): { number }
	task.desynchronize()
	local D, N = self.X._shape[1], self.X._shape[2]
	local B = Q._shape[2]
	local out = table.create(B, 1)
	for j = 1, B do
		local distIdx = table.create(N)
		for n = 1, N do
			local s = 0.0
			for i = 1, D do
				local a = self.X._storage[(i - 1) * N + n]
				local b = Q._storage[(i - 1) * B + j]
				local d = a - b
				s += d * d
			end
			distIdx[n] = { s, n }
		end
		table.sort(distIdx, function(a, b)
			return a[1] < b[1]
		end)
		local votes = {}
		local K = math.min(self.k, N)
		for r = 1, K do
			local label = self.y[distIdx[r][2]]
			votes[label] = (votes[label] or 0) + 1
		end
		local best, arg = -1, 1
		for lbl, cnt in pairs(votes) do
			if cnt > best then
				best, arg = cnt, lbl
			end
		end
		out[j] = arg
	end
	task.synchronize()
	return out
end

return KNN.new
