local utils = {}

function utils.varyContrast(input, scale)
  input = input:float()
  local mean = input:mean()
  return input:mul(scale):add((1 - scale) * mean)
end

function utils.scanContour(imgDHW, isBoundaryFunc, backward)
  local h, w = imgDHW:size(2), imgDHW:size(3)
  local filled = imgDHW.new():resize(h, w)
  local from, to, step = 1, w, 1
  if backward then
    from, to, step = w, 1, -1
  end
  for i = 1, h do
    local cnt = 0
    local isCurrentBoundary = false
    for j = from, to, step do
      if isBoundaryFunc(imgDHW[{{1,-1},i,j}]) then
        if not isCurrentBoundary then
          isCurrentBoundary = true
          cnt = cnt + 1
        end
      else
        if isCurrentBoundary then
          isCurrentBoundary = false
        end
      end
      filled[i][j] = cnt
    end
  end
  filled = filled % 2
  return filled
end

function utils.drawImage(fileName, raw2D, label)
  local w, h = raw2D:size(1), raw2D:size(2)
  local img = raw2D.new():resize(3, w, h):zero()
  img[1]:copy(raw2D)
  if label then
    img[2]:copy(raw2D)
    img[3]:copy(raw2D)
    local yellowMask = label:eq(1)
    img[1]:maskedFill(yellowMask, 255)
    img[2]:maskedFill(yellowMask, 255)
    img[3]:maskedFill(yellowMask, 0)
    local redMask = label:eq(2)
    img[1]:maskedFill(redMask, 255)
    img[2]:maskedFill(redMask, 0)
    img[3]:maskedFill(redMask, 0)
  end
  image.save(fileName, img)
end

return utils
