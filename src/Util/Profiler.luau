--!optimize 2
local Types = require("../Types")
local Profiler = {}

local fmt = string.format
local stats: { [string]: Types.Stat } = {}
local stack: { Types.Marker } = {}
local enabled: boolean = true

local function ensureStat(name: string): Types.Stat
	local stat = stats[name]
	if not stat then
		stat = {
			name = name,
			total = 0,
			selfTime = 0,
			count = 0,
			min = math.huge,
			max = 0,
			last = 0,
		}
		stats[name] = stat
	end
	return stat
end

local function clamp(value: number): number
	if value < 0 or value ~= value then
		return 0
	end
	return value
end

local function formatTime(value: number, precision: number): string
	local absValue, unit, scaled = math.abs(value)
	if absValue >= 1 then
		unit = "s"
		scaled = value
	elseif absValue >= 1e-3 then
		unit = "ms"
		scaled = value * 1e3
	elseif absValue >= 1e-6 then
		unit = "us"
		scaled = value * 1e6
	else
		unit = "ns"
		scaled = value * 1e9
	end
	local prec = math.max(0, precision)
	local spec = "%." .. tostring(prec) .. "f%s"
	return fmt(spec, scaled, unit)
end

function Profiler.reset(): ()
	table.clear(stats)
	table.clear(stack)
end

function Profiler.flush(): ()
	table.clear(stack)
end

function Profiler.isEnabled(): boolean
	return enabled
end

function Profiler.setEnabled(flag: boolean): ()
	if flag == enabled then
		return
	end
	enabled = flag
	if not enabled then
		table.clear(stack)
	end
end

function Profiler.start(name: string): ()
	if not enabled then
		return
	end
	table.insert(stack, {
		name = name,
		start = os.clock(),
		child = 0,
	})
end

function Profiler.stop(name: string?): number?
	if not enabled then
		return nil
	end
	local depth = #stack
	if depth == 0 then
		-- error("Profiler.stop called without matching Profiler.start", 2)
		return nil
	end
	local marker = stack[depth]
	stack[depth] = nil
	if name and marker.name ~= name then
		error(`Profiler.stop mismatch (expected '{marker.name}', got '{name}')`, 2)
	end
	local elapsed = clamp(os.clock() - marker.start)
	local exclusive = clamp(elapsed - marker.child)
	local stat = ensureStat(marker.name)
	stat.total += elapsed
	stat.selfTime += exclusive
	stat.count += 1
	if elapsed < stat.min then
		stat.min = elapsed
	end
	if elapsed > stat.max then
		stat.max = elapsed
	end
	stat.last = elapsed
	if depth > 1 then
		stack[depth - 1].child += elapsed
	end
	return elapsed
end

local function callWithProfiling(name: string, fn: Types.Callable, ...: any): (number, ...any)
	assert(type(fn) == "function", "Profiler: expected callable")
	if not enabled then
		local results = table.pack(pcall(fn, ...))
		if not results[1] then
			error(results[2], 0)
		end
		return 0, table.unpack(results, 2, results.n)
	end
	local packed = table.pack(pcall(function(...)
		Profiler.start(name)
		return fn(...)
	end, ...))
	local elapsed = Profiler.stop(name) or 0
	if not packed[1] then
		error(packed[2], 0)
	end
	return elapsed, table.unpack(packed, 2, packed.n)
end

function Profiler.scope(name: string, fn: Types.Callable, ...: any): (number, ...any)
	return callWithProfiling(name, fn, ...)
end

function Profiler.wrap(fn: Types.Callable, name: string?)
	assert(type(fn) == "function", "Profiler.wrap expects a function")
	local label = name or "Profiler.wrap"
	return function(...)
		return select(2, callWithProfiling(label, fn, ...))
	end
end

function Profiler.instrument(target: any, methods: { string } | string, opts: { prefix: string? }?)
	assert(type(target) == "table", "Profiler.instrument expects a table target")
	local names
	if type(methods) == "string" then
		names = { methods }
	else
		names = methods
	end
	local prefix = if opts and opts.prefix then opts.prefix else tostring(target)
	local originals = {}
	for _, key in names do
		local fn = target[key]
		assert(type(fn) == "function", `Profiler.instrument: '{key}' is not a function`)
		originals[key] = fn
		target[key] = function(...)
			return select(2, callWithProfiling(`${prefix}.{key}`, fn, ...))
		end
	end
	return function()
		for key, original in originals do
			target[key] = original
		end
	end
end

local function copyStat(stat: Types.Stat, total: number): Types.Sample
	local minv = if stat.min == math.huge then 0 else stat.min
	local count = stat.count
	local avg = if count > 0 then stat.total / count else 0
	local selfAvg = if count > 0 then stat.selfTime / count else 0
	local pct = if total > 0 then (stat.total / total) * 100 else 0
	return {
		name = stat.name,
		total = stat.total,
		self = stat.selfTime,
		count = count,
		avg = avg,
		selfAvg = selfAvg,
		min = minv,
		max = stat.max,
		last = stat.last,
		percent = pct,
	}
end

local function collectTotals(): (number, number)
	local total = 0
	local n = 0
	task.desynchronize()
	for _, stat in stats do
		total += stat.total
		n += 1
	end
	task.synchronize()
	return total, n
end

function Profiler.snapshot(): ({ Types.Sample }, number)
	local total, count = collectTotals()
	local out = table.create(count)
	local i = 1
	task.desynchronize()
	for _, stat in stats do
		out[i] = copyStat(stat, total)
		i += 1
	end
	task.synchronize()
	return out, total
end

function Profiler.get(name: string): Types.Sample?
	local stat = stats[name]
	if not stat then
		return nil
	end
	local total = 0
	task.desynchronize()
	for _, s in stats do
		total += s.total
	end
	task.synchronize()
	return copyStat(stat, total)
end

type ReportOptions = {
	sortBy: ("total" | "self" | "avg" | "selfAvg" | "max" | "count" | "last" | "min" | "percent")?,
	limit: number?,
	precision: number?,
	print: boolean?,
}

local function sortEntries(entries: { Types.Sample }, key: string)
	table.sort(entries, function(a, b)
		local av = (a :: any)[key] or 0
		local bv = (b :: any)[key] or 0
		if av == bv then
			return a.name < b.name
		end
		return av > bv
	end)
end

function Profiler.report(opts: ReportOptions?): ({ Types.Sample }, number)
	local options: ReportOptions = opts or {}
	local sortBy = options.sortBy or "total"
	local precision = options.precision or 4
	local entries, total = Profiler.snapshot()
	sortEntries(entries, sortBy)
	local limit = options.limit
	if limit and limit < #entries then
		for i = limit + 1, #entries do
			entries[i] = nil
		end
	end
	if options.print ~= false then
		print(fmt("[Profiler] total %s (%d sections)", formatTime(total, precision), #entries))
		print(fmt("%-24s %12s %12s %12s %8s %12s %12s", "section", "total", "self", "avg", "count", "min", "max"))
		for _, entry in entries do
			local totalStr = formatTime(entry.total, precision)
			local selfStr = formatTime(entry.self, precision)
			local avgStr = formatTime(entry.avg, precision)
			local minStr = formatTime(entry.min, precision)
			local maxStr = formatTime(entry.max, precision)
			print(
				fmt(
					"%-24s %12s %12s %12s %8d %12s %12s (%5.1f%%)",
					entry.name,
					totalStr,
					selfStr,
					avgStr,
					entry.count,
					minStr,
					maxStr,
					entry.percent
				)
			)
		end
	end
	return entries, total
end

function Profiler.withEnabled(flag: boolean, fn: Types.Callable, ...: any): ...any
	assert(type(fn) == "function", "Profiler.withEnabled expects a function")
	local prev = enabled
	Profiler.setEnabled(flag)
	local packed = table.pack(pcall(fn, ...))
	Profiler.setEnabled(prev)
	if not packed[1] then
		error(packed[2], 0)
	end
	return table.unpack(packed, 2, packed.n)
end

Profiler.push = Profiler.start
Profiler.pop = Profiler.stop
Profiler.time = Profiler.scope

return Profiler
