require 'torch'
require 'image'

require 'fast_neural_style.DataLoader'
require 'fast_neural_style.PerceptualCriterion'

local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local models = require 'fast_neural_style.models'
local myModel = require 'models'

local modelId = "percept-morefilters-notanh"
local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'checkpoint/mri-' .. modelId .. '/5_40000.t7', 'checkpoint')
cmd:option('-task', 'transform', 'style|transform')
cmd:option('-h5_file', 'data/mri.h5')
cmd:option('-preprocessing', 'vgg')
cmd:option('-batch_size', 4)
cmd:option('-outputDir', '/home/saxiao/tmp/perceptLoss/rlt/' .. modelId)

-- Backend options
cmd:option('-gpu', 1)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')
local opt = cmd:parse(arg)

paths.mkdir(opt.outputDir)
local loader = DataLoader(opt)

local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
print(dtype)
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model:type(dtype)
local preprocess_fn = preprocess[opt.preprocessing]
print(loader.num_minibatches['val'])
local N = 4
local cnt = 0
for b=1, N do
  print(b)
  local x, y = loader:getBatch('val')
  x, y = x:type(dtype), y:type(dtype)
  print(x:min(), x:max())
  print(preprocess_fn.deprocess(x):min(), preprocess_fn.deprocess(x):max(), preprocess_fn.deprocess(x):type())
  local y_gen = model:forward(x)
  print("y_gen", y_gen:min(), y_gen:max())
  for i=1, x:size(1) do
    cnt = cnt + 1
    local x_file = string.format("%s/%d_input.png", opt.outputDir, cnt)
    image.save(x_file, preprocess_fn.deprocess(x)[i])
    local y_gen_file = string.format("%s/%d_gen.png", opt.outputDir, cnt)
    image.save(y_gen_file, preprocess_fn.deprocess(y_gen)[i])
    local y_file = string.format("%s/%d.png", opt.outputDir, cnt)
    image.save(y_file, preprocess_fn.deprocess(y)[i])    
  end
end

