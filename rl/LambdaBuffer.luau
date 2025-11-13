--!optimize 2
local LambdaBuf = {}
LambdaBuf.__index = LambdaBuf

--@param gamma number
--@param lambda number -- 0..1
function LambdaBuf.new(gamma:number, lambda:number)
	return setmetatable({ gamma=gamma, lambda=lambda, buf={} }, LambdaBuf)
end

--@param t {state:any, action:number, reward:number, nextState:any, done:boolean}
--@return { {state:any, action:number, reward:number, nextState:any, done:boolean} }
function LambdaBuf:push(t)
	local out = {}
	local b = self.buf; table.insert(b, t)
	if t.done then
		while #b > 0 do
			local G, w, k = 0, 1, 0
			task.desynchronize()
			for i=1,#b do
				k += 1
				local ti = b[i]
				G += w * (self.gamma)^(i-1) * ti.reward
				w *= self.lambda
				if ti.done then break end
			end
			task.synchronize()
			table.insert(out, { state=b[1].state, action=b[1].action, reward=G, nextState=b[k].nextState, done=true })
			table.remove(b, 1)
		end
	else
		if #b > 20 then table.remove(b, 1) end
	end
	return out
end

return LambdaBuf
