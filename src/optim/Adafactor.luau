--!native
--!optimize 2
local Types = require("../Types")

local function flatShape(shape: { number }): (number, number)
	local rank = #shape
	local rows = 1
	for i = 1, rank - 1 do
		rows *= shape[i]
	end
	return rows, shape[rank]
end

local function ensureZeros(buf: { number }?, size: number): { number }
	if not buf or #buf ~= size then
		return table.create(size, 0)
	end
	for i = 1, size do
		buf[i] = 0
	end
	return buf
end

local function ensureBuffer(buf: { number }?, size: number): { number }
	if not buf or #buf ~= size then
		return table.create(size, 0)
	end
	return buf
end

return function(
	params: { Types.Tensor },
	lr: number,
	beta2: number?,
	eps: number?,
	clipThreshold: number?,
	weightDecay: number?
)
	beta2 = beta2 or 0.999
	eps = eps or 1e-30
	clipThreshold = clipThreshold or 0
	weightDecay = weightDecay or 0

	local factored: {
		[Types.Tensor]: {
			row: { number },
			col: { number },
			rowMean: { number },
			colMean: { number },
			updates: { number },
		},
	} =
		{}
	local full: { [Types.Tensor]: { v: { number }, updates: { number } } } = {}
	local self = {}

	local function clipScaleOf(updates: { number }, count: number)
		if clipThreshold == 0 then
			return 1
		end
		local acc = 0
		for i = 1, count do
			local u = updates[i]
			acc += u * u
		end
		local rms = math.sqrt(acc / count)
		if rms > clipThreshold then
			return clipThreshold / rms
		end
		return 1
	end

	function self:step()
		task.desynchronize()
		for _, p in params do
			local g = p._grad
			if g then
				local ps, gs = p._storage, g._storage
				local count = #ps
				local shape = p._shape
				local rank = #shape
				if rank >= 2 then
					local rows, cols = flatShape(shape)
					local s = factored[p]
					if not s then
						s = {
							row = table.create(rows, 0),
							col = table.create(cols, 0),
							rowMean = table.create(rows, 0),
							colMean = table.create(cols, 0),
							updates = table.create(count, 0),
						}
						factored[p] = s
					elseif #s.row ~= rows or #s.col ~= cols then
						-- reshape
						s.row = table.create(rows, 0)
						s.col = table.create(cols, 0)
						s.rowMean = table.create(rows, 0)
						s.colMean = table.create(cols, 0)
						s.updates = table.create(rows * cols, 0)
					end
					local rowMean = ensureZeros(s.rowMean, rows)
					s.rowMean = rowMean
					local colMean = ensureZeros(s.colMean, cols)
					s.colMean = colMean
					local updates = ensureBuffer(s.updates, count)
					s.updates = updates
					for i = 1, count do
						local g2 = gs[i] * gs[i]
						local r = math.floor((i - 1) / cols) + 1
						local c = ((i - 1) % cols) + 1
						rowMean[r] += g2
						colMean[c] += g2
					end
					local invCols = 1 / cols
					local invRows = 1 / rows
					for r = 1, rows do
						rowMean[r] *= invCols
					end
					for c = 1, cols do
						colMean[c] *= invRows
					end
					local rowState = s.row
					local colState = s.col
					for r = 1, rows do
						rowState[r] = (beta2 :: number) * rowState[r] + (1 - (beta2 :: number)) * rowMean[r]
					end
					for c = 1, cols do
						colState[c] = (beta2 :: number) * colState[c] + (1 - (beta2 :: number)) * colMean[c]
					end
					local meanRow = 0
					for r = 1, rows do
						meanRow += rowState[r]
					end
					meanRow /= rows
					for i = 1, count do
						local r = math.floor((i - 1) / cols) + 1
						local c = ((i - 1) % cols) + 1
						local vHat = (rowState[r] * colState[c]) / (meanRow + (eps :: number))
						updates[i] = gs[i] / (math.sqrt(vHat) + (eps :: number))
					end
					local scale = clipScaleOf(updates, count)
					for i = 1, count do
						ps[i] -= lr * (updates[i] * scale + (weightDecay :: number) * ps[i])
					end
				else
					local v = full[p]
					if not v then
						v = { v = table.create(count, 0), updates = table.create(count, 0) }
						full[p] = v
					elseif #v.v ~= count then
						v.v = table.create(count, 0)
						v.updates = table.create(count, 0)
					end
					local updates = ensureBuffer(v.updates, count)
					v.updates = updates
					local vv = v.v
					for i = 1, count do
						local gval = gs[i]
						vv[i] = (beta2 :: number) * vv[i] + (1 - (beta2 :: number)) * gval * gval
						updates[i] = gval / (math.sqrt(vv[i]) + (eps :: number))
					end
					local scale = clipScaleOf(updates, count)
					for i = 1, count do
						ps[i] -= lr * (updates[i] * scale + (weightDecay :: number) * ps[i])
					end
				end
			end
		end
		task.synchronize()
	end
	function self:zeroGrad()
		for _, p in params do
			p._grad = nil
		end
	end
	function self:setLr(newLr: number)
		lr = newLr
	end
	function self:getState()
		return {
			lr = lr,
			beta2 = beta2,
			eps = eps,
			clipThreshold = clipThreshold,
			weightDecay = weightDecay,
			factored = factored,
			full = full,
		}
	end
	function self:setState(state: {
		lr: number,
		beta2: number,
		eps: number,
		clipThreshold: number,
		weightDecay: number,
		factored: { [Types.Tensor]: { row: { number }, col: { number }, rowMean: { number }, colMean: { number }, updates: { number } } },
		full: { [Types.Tensor]: { v: { number }, updates: { number } } },
	})
		if state.lr ~= nil then
			lr = state.lr
		end
		if state.beta2 ~= nil then
			beta2 = state.beta2
		end
		if state.eps ~= nil then
			eps = state.eps
		end
		if state.clipThreshold ~= nil then
			clipThreshold = state.clipThreshold
		end
		if state.weightDecay ~= nil then
			weightDecay = state.weightDecay
		end
		if state.factored ~= nil then
			factored = state.factored
		end
		if state.full ~= nil then
			full = state.full
		end
	end
	return self
end
