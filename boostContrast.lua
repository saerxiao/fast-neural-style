require 'image'

local utils = require 'utils'

local dir = "/home/saxiao/tmp/mri/mip/percept-unet"
local picId = "IXI021_N"
local fileName = string.format("%s/%s_gen.png", dir, picId)
local img = image.load(fileName)
print(img:min(), img:max())

--img = img:mul(1/img:max())
img = utils.varyContrast(img, 2)
img:maskedFill(img:lt(0), 0)
fileName = string.format("%s/%s_gen_bc.png", dir, picId)
image.save(fileName, img)



