--!optimize 2
local Tensor = {}
Tensor.__index = Tensor

local Util = require("./Util")
local Types = require("./Types")
local Tape -- lazy-load

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.zeros(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Types.Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	local self = setmetatable({
		_storage = table.create(n, 0),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
end

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.ones(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?): Types.Tensor
	dtype = dtype or "f64"
	local n = Util.sizeFromShape(shape)
	local self = setmetatable({
		_storage = table.create(n, 1),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
end

--@param data {number}
--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.fromArray(
	data: { number },
	shape: { number },
	dtype: Types.DType?,
	requiresGrad: boolean?
): Types.Tensor
	local n = Util.sizeFromShape(shape)
	assert(#data == n, ("Tensor.fromArray: got %d elements, expected %d"):format(#data, n))
	dtype = dtype or "f64"
	local self = setmetatable({
		_storage = table.clone(data),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
end

--@param self Tensor
--@param newShape {number}
--@return Tensor
function Tensor.reshape(self: Types.Tensor, newShape: { number }): Types.Tensor
	assert(Util.sizeFromShape(newShape) == #self._storage, "reshape: element count mismatch")
	local out = setmetatable({
		_storage = self._storage,
		_shape = table.clone(newShape),
		_dtype = self._dtype,
		_requiresGrad = self._requiresGrad,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)

	if not self._requiresGrad then
		return out
	end
	if not Tape then
		Tape = require("./autograd/Tape") :: any
	end
	return Tape.makeNode(out, { self }, function(gy: Types.Tensor)
		if self._requiresGrad then
			self._grad = self._grad or Tensor.zeros(self._shape, self._dtype, false)
			task.desynchronize()
			local gs, ys = self._grad._storage, gy._storage
			for i = 1, #ys do
				gs[i] += ys[i]
			end
			task.synchronize()
		end
	end)
end

function Tensor.detach(self: Types.Tensor): Types.Tensor
	local out = setmetatable({
		_storage = self._storage,
		_shape = table.clone(self._shape),
		_dtype = self._dtype,
		_requiresGrad = false,
		_parents = nil,
		_backprop = nil,
		_grad = nil,
	}, Tensor)
	return out
end

return Tensor
