--!native
--!optimize 2
local function count(model): (number, number)
	local ps = model:parameters()
	local total, bytes = 0, 0
	task.desynchronize()
	for i = 1, #ps do
		local t = ps[i]
		local n = #t._storage
		total += n
		bytes += n * 8 -- 8 bytes
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
