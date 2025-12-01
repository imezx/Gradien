--!native
--!optimize 2
local Split = {}
Split.__index = Split

type Behavior = "Removed" | "Isolated" | "MergedWithPrevious" | "MergedWithNext"

function Split.new(pattern: string, behavior: Behavior, invert: boolean?)
	local self = setmetatable({}, Split)
	self.pattern = pattern
	self.behavior = behavior or "Removed"
	self.invert = invert or false
	return self
end

function Split:pre_tokenize(text: string)
	local splits = {}
	local offsets = {}

	if self.invert then
		for s, e in string.gmatch(text, "()" .. self.pattern .. "()") do
			table.insert(splits, string.sub(text, s, e - 1))
			table.insert(offsets, { s, e - 1 })
		end
		return splits, offsets
	end

	local last_pos = 1
	for s, e in string.gmatch(text, "()" .. self.pattern .. "()") do
		local pre_segment = string.sub(text, last_pos, s - 1)
		local separator = string.sub(text, s, e - 1)

		if self.behavior == "Removed" then
			if #pre_segment > 0 then
				table.insert(splits, pre_segment)
				table.insert(offsets, { last_pos, s - 1 })
			end
			last_pos = e
		elseif self.behavior == "Isolated" then
			if #pre_segment > 0 then
				table.insert(splits, pre_segment)
				table.insert(offsets, { last_pos, s - 1 })
			end
			table.insert(splits, separator)
			table.insert(offsets, { s, e - 1 })
			last_pos = e
		elseif self.behavior == "MergedWithPrevious" then
			if #pre_segment > 0 then
				table.insert(splits, pre_segment .. separator)
				table.insert(offsets, { last_pos, e - 1 })
			else
				table.insert(splits, separator)
				table.insert(offsets, { s, e - 1 })
			end
			last_pos = e
		elseif self.behavior == "MergedWithNext" then
			if #pre_segment > 0 then
				table.insert(splits, pre_segment)
				table.insert(offsets, { last_pos, s - 1 })
			end
			last_pos = s
		end
	end
	if last_pos <= #text then
		table.insert(splits, string.sub(text, last_pos))
		table.insert(offsets, { last_pos, #text })
	end
	return splits, offsets
end

return Split
