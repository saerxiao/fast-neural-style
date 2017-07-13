local utils = {}

function utils.varyContrast(input, scale)
  input = input:float()
  local mean = input:mean()
  return input:mul(scale):add((1 - scale) * mean)
end

return utils
