require 'image'

local utils = require 'utils'

local dataDir = "/data/mri/data/multi/valid"
local volId = "IXI291"
local sliceId = "065"
local imgFile = string.format("%s/%s-%s.png", dataDir, volId, sliceId)
local img = image.load(imgFile):mul(255)

print(img:min(), img:max())
local center = {x=256/2, y=256/2}
local outputdir = "/home/saxiao/tmp/mri/skull"
paths.mkdir(outputdir)

local threshold = 10
local function isBoundaryFunc(dhw)
  if dhw[1] >  threshold or dhw[2] > threshold or dhw[3] > threshold then
    return false
  else
    return true
  end
end

local filled = utils.scanContour(img, isBoundaryFunc)
print(filled:sum())
local plotName = string.format("%s/%s-%s.png", outputdir, volId, sliceId)
utils.drawImage(plotName, img[1], filled)
