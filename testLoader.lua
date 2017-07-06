require 'fast_neural_style.DataLoader'

local cmd = torch.CmdLine()
cmd:option('-h5_file', 'data/toy.h5')
cmd:option('-task', 'transform', 'style|upsample|transform')
cmd:option('-preprocessing', 'vgg')
cmd:option('-batch_size', 4)
local opt = cmd:parse(arg)

local loader = DataLoader(opt)
local N = 5
for i=1, N do
    local x, y = loader:getBatch('train')
    print(x:size(), y:size())
end

