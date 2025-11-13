--!optimize 2
local Tensor = {}
Tensor.__index = Tensor

local Util = require("./Util")
local Types = require("./Types")

--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.zeros(shape: { number }, dtype: Types.DType?, requiresGrad: boolean?)
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

--@param data {number}
--@param shape {number}
--@param dtype string?
--@param requiresGrad boolean?
--@return Tensor
function Tensor.fromArray(data: { number }, shape: { number }, dtype: Types.DType?, requiresGrad: boolean?)
	dtype = dtype or "f64"
	local self = setmetatable({
		_storage = table.clone(data),
		_shape = table.clone(shape),
		_dtype = dtype,
		_requiresGrad = requiresGrad == true,
	}, Tensor)
	return self
end

return Tensor
