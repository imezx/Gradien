--!native
--!optimize 2
local Tape = require("../autograd/Tape")
local Tensor = require("../Tensor")
local Types = require("../Types")
local Util = require("../Util")

local assert = Util.Assert

local A = {}

local function prep(x: Types.Tensor): Types.Tensor
	return Tensor.zeros(x._shape, x._requiresGrad)
end

local function ensureGrad(x: Types.Tensor): Types.Tensor?
	if not x._grad then
		x._grad = Tensor.zeros(x._shape)
	end
	return x._grad
end

function A.ReLU(x: Types.Tensor): Types.Tensor
	return Tape.relu(x)
end

function A.Sigmoid(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		ys[i] = 1 / (1 + math.exp(-xs[i]))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #ys do
			local s = ys[i]
			gxs[i] += gys[i] * s * (1 - s)
		end
		task.synchronize()
	end)
end

function A.Tanh(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		ys[i] = math.tanh(xs[i])
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #ys do
			local t = ys[i]
			gxs[i] += gys[i] * (1 - t * t)
		end
		task.synchronize()
	end)
end

function A.GELU(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	local c = math.sqrt(2 / math.pi)
	local k = 0.044715

	for i = 1, #xs do
		local v = xs[i]
		local cube = v * v * v
		local inner = c * (v + k * cube)
		ys[i] = 0.5 * v * (1 + math.tanh(inner))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		local c2 = math.sqrt(2 / math.pi)

		for i = 1, #xs do
			local v = xs[i]
			local v2 = v * v
			local inner = c2 * (v + 0.044715 * v * v2)
			local t = math.tanh(inner)
			local sech2 = 1 - t * t
			local d_inner = c2 * (1 + 3 * 0.044715 * v2)

			local d = 0.5 * (1 + t) + 0.5 * v * sech2 * d_inner
			gxs[i] += gys[i] * d
		end
		task.synchronize()
	end)
end

function A.LeakyReLU(x: Types.Tensor, alpha: number?): Types.Tensor
	local a = alpha or 0.01
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		local v = xs[i]
		ys[i] = (v > 0) and v or (a * v)
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #xs do
			local v = xs[i]
			local slope = (v > 0) and 1 or a
			gxs[i] += gys[i] * slope
		end
		task.synchronize()
	end)
end

function A.Softplus(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		ys[i] = math.log(1 + math.exp(xs[i]))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #xs do
			local v = xs[i]
			local s = 1 / (1 + math.exp(-v)) -- sigmoid
			gxs[i] += gys[i] * s
		end
		task.synchronize()
	end)
end

function A.ELU(x: Types.Tensor, alpha: number?): Types.Tensor
	local a = alpha or 1.0
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		local v = xs[i]
		ys[i] = (v > 0) and v or (a * (math.exp(v) - 1))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #xs do
			local v = xs[i]
			local d = (v > 0) and 1 or (ys[i] + a)
			gxs[i] += gys[i] * d
		end
		task.synchronize()
	end)
end

function A.SiLU(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		local v = xs[i]
		local s = 1 / (1 + math.exp(-v))
		ys[i] = v * s
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #xs do
			local v = xs[i]
			local s = 1 / (1 + math.exp(-v))
			local d = s + v * s * (1 - s)
			gxs[i] += gys[i] * d
		end
		task.synchronize()
	end)
end

function A.Mish(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	local t = 20 -- threshold
	for i = 1, #xs do
		local v = xs[i]
		if v > t then
			-- softplus(x)
			ys[i] = v * math.tanh(v)
		else
			local sp = math.log(1 + math.exp(v))
			ys[i] = v * math.tanh(sp)
		end
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		for i = 1, #xs do
			local v = xs[i]
			local d
			if v > t then
				local tanh_v = math.tanh(v)
				d = tanh_v + v * (1 - tanh_v * tanh_v)
			else
				local exp_v = math.exp(v)
				local sp = math.log(1 + exp_v)
				local tanh_sp = math.tanh(sp)
				local sigmoid = 1 / (1 + math.exp(-v))
				d = tanh_sp + v * (1 - tanh_sp * tanh_sp) * sigmoid
			end
			gxs[i] += gys[i] * d
		end
		task.synchronize()
	end)
end

function A.SeLU(x: Types.Tensor, alpha: number?, lambda: number?): Types.Tensor
	local a = alpha or 1.6732632423543772
	local l = lambda or 1.0507009873554805
	task.desynchronize()
	local y = prep(x)
	local xs, ys = x._storage, y._storage
	for i = 1, #xs do
		local v = xs[i]
		ys[i] = (v > 0) and (l * v) or (l * a * (math.exp(v) - 1))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs, gys = gx._storage, gy._storage
		local la = l * a
		for i = 1, #xs do
			local v = xs[i]
			local d = (v > 0) and l or (la * math.exp(v))
			gxs[i] += gys[i] * d
		end
		task.synchronize()
	end)
end

function A.SwiGLU(a: Types.Tensor, b: Types.Tensor): Types.Tensor
	assert(#a._storage == #b._storage, "SwiGLU: size mismatch")
	task.desynchronize()
	local y = Tensor.zeros(a._shape, a._requiresGrad or b._requiresGrad)
	local as, bs, ys = a._storage, b._storage, y._storage
	for i = 1, #as do
		local val_a = as[i]
		local s = 1 / (1 + math.exp(-val_a))
		ys[i] = (val_a * s) * bs[i]
	end
	task.synchronize()
	return Tape.makeNode(y, { a, b }, function(gy)
		if not (a._requiresGrad or b._requiresGrad) then
			return
		end

		local ga = a._requiresGrad and ensureGrad(a)
		local gb = b._requiresGrad and ensureGrad(b)

		task.desynchronize()
		local gas = ga and ga._storage
		local gbs = gb and gb._storage
		local gys = gy._storage

		for i = 1, #as do
			local val_a = as[i]
			local val_b = bs[i]
			local g = gys[i]

			local s = 1 / (1 + math.exp(-val_a))
			local swish = val_a * s

			-- d/da (swish * b) = b * (swish + x * s * (1-s)) -> actually d/dx (x*s) = s + x*s*(1-s)
			local d_swish = s + val_a * s * (1 - s)

			if gas then
				gas[i] += g * (val_b * d_swish)
			end
			if gbs then
				gbs[i] += g * swish
			end
		end
		task.synchronize()
	end)
end

function A.SwiGLUSplit(x: Types.Tensor, hidden: number?): Types.Tensor
	local rows, B = x._shape[1], x._shape[2]
	local H = math.floor(rows / 2)
	assert(rows == 2 * H, "SwiGLUSplit: first dim must be even")
	if hidden then
		assert(hidden == H, "SwiGLUSplit: hidden mismatch")
	end

	task.desynchronize()
	local y = Tensor.zeros({ H, B }, x._requiresGrad)
	local xs, ys = x._storage, y._storage
	for j = 1, B do
		-- a is at x[(0...H-1)*B + j]
		-- b is at x[(H...2H-1)*B + j]
		for i = 1, H do
			local idx_a = (i - 1) * B + j
			local idx_b = (i + H - 1) * B + j

			local val_a = xs[idx_a]
			local val_b = xs[idx_b]

			local s = 1 / (1 + math.exp(-val_a))
			ys[(i - 1) * B + j] = (val_a * s) * val_b
		end
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then
			return
		end
		local gx = ensureGrad(x)
		task.desynchronize()
		local gxs = gx._storage
		local gys = gy._storage
		for j = 1, B do
			for i = 1, H do
				local idx_a = (i - 1) * B + j
				local idx_b = (i + H - 1) * B + j
				local idx_y = (i - 1) * B + j

				local val_a = xs[idx_a]
				local val_b = xs[idx_b]
				local g = gys[idx_y]

				local s = 1 / (1 + math.exp(-val_a))
				local swish = val_a * s
				local d_swish = s + val_a * s * (1 - s)

				gxs[idx_a] += g * (val_b * d_swish)
				gxs[idx_b] += g * swish
			end
		end
		task.synchronize()
	end)
end

return A
