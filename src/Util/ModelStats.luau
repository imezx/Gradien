--!optimize 2
local function bytesPer(dtype: string?): number
	if dtype == "f64" then return 8 end
	if dtype == "u8" then return 1 end
	return 4
end

local function count(model): (number, number)
	local ps = model:parameters()
	local total, bytes = 0, 0
	task.desynchronize()
	for i = 1, #ps do
		local t = ps[i]
		local n = #t._storage
		total += n
		bytes += n * bytesPer(t._dtype)
	end
	task.synchronize()
	return total, bytes
end

local function fmtBytes(n: number): string
	local units = { "B", "KB", "MB", "GB", "TB" }
	local i = 1
	while n >= 1024 and i < #units do
		n /= 1024; i += 1
	end
	return string.format("%.2f %s", n, units[i])
end

return {
	count = count,
	formatBytes = fmtBytes,
	summary = function(model, printOut: boolean?): (number, string)
		local p, b = count(model)
		local fmt = fmtBytes(b)
		if printOut then
			print(`params={p} (~{fmt})`)
		end
		return p, fmt
	end,
}
