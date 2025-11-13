--!optimize 2
local Hooks = {}

function Hooks.addForwardHook(module, fn)
	local orig = module.forward
	module.___fwd = orig
	module.forward = function(self, ...)
		local out = orig(self, ...)
		fn(self, out)
		return out
	end
end

function Hooks.removeForwardHook(module)
	if module.___fwd then
		module.forward = module.___fwd
		module.___fwd = nil
	end
end

return Hooks
