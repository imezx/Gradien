--!optimize 2
local Types = require("./Types")
local GradClip = require("./GradClip")
local Anomaly = require("./Util/Anomaly")

type Tensor = Types.Tensor

local Debug = {}

function Debug.checkTensor(t: Tensor, name: string?): boolean
	if Anomaly.hasBadValues(t) then
		warn(`Gradien.Debug: tensor '{name or "?"}' has NaN/Inf/invalid values`)
		return false
	end
	return true
end

function Debug.checkGradients(params: { Tensor }, label: string?): (boolean, number?)
	for i, p in params do
		local g = p._grad
		if g and Anomaly.hasBadValues(g) then
			warn(`Gradien.Debug: bad gradient in param {i} ({label or ""})`)
			return false, i
		end
	end
	return true, nil
end

function Debug.wrapOptimizer(
	opt,
	params: { Tensor },
	cfg: { maxGradNorm: number?, clipValue: number?, warnOnNaN: boolean?, label: string? }?
)
	cfg = cfg or {}
	local maxGradNorm = cfg.maxGradNorm
	local clipValue = cfg.clipValue
	local warnOnNaN = cfg.warnOnNaN
	if type(warnOnNaN) ~= "boolean" then
		warnOnNaN = true
	end
	local label = cfg.label or "wrapped optimizer"

	local wrapped = {}

	function wrapped:zeroGrad()
		opt:zeroGrad()
	end

	function wrapped:step()
		if maxGradNorm then
			GradClip.clipNorm(params, maxGradNorm)
		end
		if clipValue then
			GradClip.clipValue(params, clipValue)
		end
		if warnOnNaN then
			Debug.checkGradients(params, label)
		end
		opt:step()
	end

	return wrapped
end

function Debug.checkModel(model, label: string?)
	local params = model:parameters()
	Debug.checkGradients(params, label)
	for i, p in params do
		Debug.checkTensor(p, `{label or "model"}.param{i}`)
	end
end

return Debug
