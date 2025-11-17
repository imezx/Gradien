--!optimize 2
--@Eternity_Devs
return function<T>(condition: T, errorMessage: string, level: number?): T
	if not (condition :: any) then
		error(errorMessage, level or 2)
	end
	return condition
end
