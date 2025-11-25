--!native
--!optimize 2
local Types = require("../Types")
return function(mods: { Types.Module<Types.Tensor, Types.Tensor> }): Types.Module<Types.Tensor, Types.Tensor>
	return {
		_layers = mods,
		forward = function(_, x: Types.Tensor): Types.Tensor
			local v = x
			for _, m in ipairs(mods) do
				if type(m) == "table" and m.forward then
					v = m:forward(v)
				elseif type(m) == "function" then
					v = m(nil, v)
				else
					error("Sequential: unsupported module entry")
				end
			end
			return v
		end,
		parameters = function(_): { Types.Tensor }
			local out = {}
			for _, m in ipairs(mods) do
				if type(m) == "table" and m.parameters then
					for _, p in ipairs(m:parameters()) do
						table.insert(out, p)
					end
				end
			end
			return out
		end,
	}
end
