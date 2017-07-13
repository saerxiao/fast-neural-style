require 'fast_neural_style.DataLoader'
require 'image'
local preprocess = require 'fast_neural_style.preprocess'

local prepModel = 'vgg'
local preprocess_fn = preprocess[prepModel]

local cmd = torch.CmdLine()
cmd:option('-h5_file', 'data/mri.h5')
cmd:option('-task', 'transform', 'style|upsample|transform')
cmd:option('-preprocessing', 'vgg')
cmd:option('-batch_size', 4)
cmd:option('-selectChannel', 2)
local opt = cmd:parse(arg)

local outputDir = "/home/saxiao/tmp/perceptLoss"
local loader = DataLoader(opt)
local N = 1
for i=1, N do
    local x, y = loader:getBatch('train')
    print("x", x:size())
    print("y", y[1]:max(), y[2]:max(), y[3]:max())
    local xfile = string.format("%s/%d_c%d_x.png", outputDir, i, opt.selectChannel)
    image.save(xfile, preprocess_fn.deprocess(x)[1])
    local yfile = string.format("%s/%d_y.png", outputDir, i)    
    image.save(yfile, preprocess_fn.deprocess(y)[1])
    print("x", x:type())
    print("x", x:min(), x:max())
    print("y", y:type())
    print("y", y:min(), y:max())
end

local function loadRawImage()
  local raw = image.load("/data/mri/data/multi/valid/IXI284-092.png")
  print("raw", raw:type(), raw:min(), raw:max())
  local mask = image.load("/data/mri/data/multi/valid/IXI284-092_mask.png")
  print("mask", mask:type(), mask:min(), mask:max())
end

