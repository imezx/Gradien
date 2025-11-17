--!optimize 2
local Tape = require("../autograd/Tape")
local Tensor = require("../Tensor")
local Types = require("../Types")
local Util = require("../Util")

local assert = Util.Assert

local A = {}

--@param x @Tensor
--@return Tensor
function A.ReLU(x: Types.Tensor): Types.Tensor
	return Tape.relu(x)
end

--@param x @Tensor
--@return Tensor
function A.Sigmoid(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		y._storage[i] = 1 / (1 + math.exp(-x._storage[i]))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			task.desynchronize()
			if not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			for i = 1, #y._storage do
				local s = y._storage[i]
				x._grad._storage[i] += gy._storage[i] * s * (1 - s)
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@return Tensor
function A.Tanh(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		y._storage[i] = math.tanh(x._storage[i])
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			task.desynchronize()
			if not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			for i = 1, #y._storage do
				local t = y._storage[i]
				x._grad._storage[i] += gy._storage[i] * (1 - t * t)
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@return Tensor
function A.GELU(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	local c = math.sqrt(2 / math.pi)
	for i = 1, #x._storage do
		local v = x._storage[i]
		local t = c * (v + 0.044715 * v * v * v)
		y._storage[i] = 0.5 * v * (1 + math.tanh(t))
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			task.desynchronize()
			if not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			local c2 = math.sqrt(2 / math.pi)
			for i = 1, #x._storage do
				local v = x._storage[i]
				local t = c2 * (v + 0.044715 * v * v * v)
				local dt = c2 * (1 + 3 * 0.044715 * v * v)
				local sech2 = 1 - math.tanh(t) ^ 2
				x._grad._storage[i] += gy._storage[i] * (0.5 * (1 + math.tanh(t)) + 0.5 * v * sech2 * dt)
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@param alpha number (default 0.01)
function A.LeakyReLU(x: Types.Tensor, alpha: number): Types.Tensor
	alpha = alpha or 0.01
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		local v = x._storage[i]
		y._storage[i] = (v > 0) and v or (alpha * v)
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			task.desynchronize()
			if not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			for i = 1, #x._storage do
				local v = x._storage[i]
				local slope = (v > 0) and 1 or alpha
				x._grad._storage[i] += gy._storage[i] * slope
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@param alpha number? (default 0.01)
--@return Tensor
function A.ELU(x: Types.Tensor, alpha: number?): Types.Tensor
	alpha = alpha or 1.0
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		local v = x._storage[i]
		if v > 0 then
			y._storage[i] = v
		else
			y._storage[i] = alpha * (math.exp(v) - 1)
		end
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			task.desynchronize()
			if not x._grad then
				x._grad = Tensor.zeros(x._shape)
			end
			for i = 1, #x._storage do
				local v = x._storage[i]
				local gradv
				if v > 0 then
					gradv = 1
				else
					-- d/dx alpha*(e^x - 1) = alpha*e^x
					-- Using y + alpha is numerically equivalent: alpha*e^x = (y + alpha)
					gradv = (y._storage[i] + alpha)
				end
				x._grad._storage[i] += gy._storage[i] * gradv
			end
			task.synchronize()
		end
	end)
end

--@param a @Tensor
--@param b @Tensor
--@return Tensor
function A.SwiGLU(a: Types.Tensor, b: Types.Tensor): Types.Tensor
	assert(a._shape[1] == b._shape[1] and a._shape[2] == b._shape[2], "SwiGLU: a and b shapes must match")
	task.desynchronize()
	local y = Tensor.zeros(a._shape, a._dtype, a._requiresGrad or b._requiresGrad)
	for i = 1, #a._storage do
		local av = a._storage[i]
		local bv = b._storage[i]
		local s = 1 / (1 + math.exp(-av)) -- sigmoid(a)
		local sw = av * s -- swish(a)
		y._storage[i] = sw * bv
	end
	task.synchronize()
	return Tape.makeNode(y, { a, b }, function(gy)
		-- dy/da = b * swish'(a), swish'(a) = s + a*s*(1-s)
		-- dy/db = swish(a)
		if a._requiresGrad or b._requiresGrad then
			task.desynchronize()
			if a._requiresGrad and not a._grad then a._grad = Tensor.zeros(a._shape) end
			if b._requiresGrad and not b._grad then b._grad = Tensor.zeros(b._shape) end
			for i = 1, #a._storage do
				local av = a._storage[i]
				local bv = b._storage[i]
				local s = 1 / (1 + math.exp(-av))
				local sw = av * s
				local swp = s + av * s * (1 - s) -- swish'(a)
				local g = gy._storage[i]
				if a._requiresGrad then
					a._grad._storage[i] += g * (bv * swp)
				end
				if b._requiresGrad then
					b._grad._storage[i] += g * sw
				end
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@return Tensor
function A.SiLU(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		local v = x._storage[i]
		local s = 1 / (1 + math.exp(-v))
		y._storage[i] = v * s
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			if not x._grad then x._grad = Tensor.zeros(x._shape) end
			task.desynchronize()
			for i = 1, #x._storage do
				local v = x._storage[i]
				local s = 1 / (1 + math.exp(-v))
				local swp = s + v * s * (1 - s) -- d/dx [x*sigmoid(x)]
				x._grad._storage[i] += gy._storage[i] * swp
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@return Tensor
function A.Mish(x: Types.Tensor): Types.Tensor
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		local v = x._storage[i]
		local sp = math.log(1 + math.exp(v)) -- softplus
		y._storage[i] = v * math.tanh(sp)
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			if not x._grad then x._grad = Tensor.zeros(x._shape) end
			task.desynchronize()
			for i = 1, #x._storage do
				local v = x._storage[i]
				local sp = math.log(1 + math.exp(v))
				local t = math.tanh(sp)
				local s = 1 / (1 + math.exp(-v)) -- d(sp)/dx = sigmoid(x)
				local dy = t + v * (1 - t * t) * s
				x._grad._storage[i] += gy._storage[i] * dy
			end
			task.synchronize()
		end
	end)
end

--@param x @Tensor
--@param alpha number? (default 1.6732632423543772)
--@param lambda number? (default 1.0507009873554805)
--@return Tensor
function A.SeLU(x: Types.Tensor, alpha: number?, lambda: number?): Types.Tensor
	alpha = alpha or 1.6732632423543772
	lambda = lambda or 1.0507009873554805
	task.desynchronize()
	local y = Tensor.zeros(x._shape, x._dtype, x._requiresGrad)
	for i = 1, #x._storage do
		local v = x._storage[i]
		if v > 0 then
			y._storage[i] = lambda * v
		else
			y._storage[i] = lambda * (alpha * (math.exp(v) - 1))
		end
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if x._requiresGrad then
			if not x._grad then x._grad = Tensor.zeros(x._shape) end
			task.desynchronize()
			for i = 1, #x._storage do
				local v = x._storage[i]
				local g = (v > 0) and lambda or (lambda * alpha * math.exp(v))
				x._grad._storage[i] += gy._storage[i] * g
			end
			task.synchronize()
		end
	end)
end

--@param x Types.Tensor @shape {2H,B}
--@param hidden number? @optional expected H; validated if provided
--@return Types.Tensor @shape {H,B}
function A.SwiGLUSplit(x: Types.Tensor, hidden: number?): Types.Tensor
	local rows, B = x._shape[1], x._shape[2]
	local H = math.floor(rows / 2)
	assert(rows == 2 * H, "SwiGLUSplit: first dim must be even (2*H)")
	if hidden then assert(hidden == H, "SwiGLUSplit: hidden mismatch") end
	task.desynchronize()
	local y = Tensor.zeros({ H, B }, x._dtype, x._requiresGrad)
	for j = 1, B do
		for i = 1, H do
			local ai = (i - 1) * B + j
			local bi = (H + i - 1) * B + j
			local a = x._storage[ai]
			local b = x._storage[bi]
			local s = 1 / (1 + math.exp(-a))
			local sw = a * s
			y._storage[(i - 1) * B + j] = sw * b
		end
	end
	task.synchronize()
	return Tape.makeNode(y, { x }, function(gy)
		if not x._requiresGrad then return end
		if not x._grad then x._grad = Tensor.zeros(x._shape) end
		task.desynchronize()
		for j = 1, B do
			for i = 1, H do
				local ai = (i - 1) * B + j
				local bi = (H + i - 1) * B + j
				local a = x._storage[ai]
				local b = x._storage[bi]
				local s = 1 / (1 + math.exp(-a))
				local sw = a * s
				local swp = s + a * s * (1 - s)
				local g = gy._storage[(i - 1) * B + j]
				x._grad._storage[ai] += g * (b * swp) -- d/da
				x._grad._storage[bi] += g * sw -- d/db
			end
		end
		task.synchronize()
	end)
end

return A
