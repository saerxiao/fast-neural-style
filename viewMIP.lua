require 'image'
require 'lfs'

local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local models = require 'fast_neural_style.models'
local myModel = require 'models'

local preprocess_fn = preprocess.vgg

local modelId = 'percept-notanh-gan-burn100'
local gpu = 1
local backend = "cuda"
local use_cudnn = 1
local checkpointFile = "checkpoint/mri-" .. modelId .. "/25_170000.t7"  -- 5_40000.t7, 11_80000.t7

local dir = "/data/mri/data/multi/valid"
--local selectVolId = 'IXI291'
local dtype, use_cudnn = utils.setup_gpu(gpu, backend, use_cudnn == 1)
local checkpoint = torch.load(checkpointFile)
local model = checkpoint.model:type(dtype)

local batchsize = 4
local plotSlice = false
local outputdir = "/home/saxiao/tmp/mri/mip/" .. modelId .. "-iter170000"
if selectVolId then
  outputdir = outputdir .. '-' .. selectVolId
end
paths.mkdir(outputdir)

local function getVolumnMap()
  local t = {}
  for file in lfs.dir(dir) do
    if string.match(file, 'png') and not string.match(file, "mask") then
      local volId = string.match(file, "(.*)-.*")
      local slices = t[volId]
      if not slices then
        slices = {}
        t[volId] = slices
      end
      table.insert(slices, file)
    end
  end
  return t
end

local function compareIdx(a, b)
  local aId = tonumber(string.match(a, ".*-(.*).png"))
  local bId = tonumber(string.match(b, ".*-(.*).png"))
  return aId < bId
end

local function transform(slices, prefix)
  table.sort(slices, compareIdx)
  local input = torch.Tensor():resize(#slices, 3, 256, 256)
  local gtrueth = torch.Tensor():resizeAs(input)
  --print(#slices)
  for i=1, #slices do
    input[i] = image.load(string.format("%s/%s", dir, slices[i]))
    local sliceId = tonumber(string.match(slices[i], ".*-(.*).png"))
    local tfile = string.format("%s/%s_mask.png", dir, slices[i]:sub(1, -5))
    local mask = image.load(tfile)
    gtrueth[i][1]:copy(mask[1])
    gtrueth[i][2]:copy(mask[1])
    gtrueth[i][3]:copy(mask[1])
  end
  --print(input:min(), input:max())
  input = preprocess_fn.preprocess(input:float()):type(dtype)
  --print(input:min(), input:max())
  local output_gen = input.new():resizeAs(input)
  local bstart = 1
  while bstart <= #slices do
    local bend = bstart + batchsize
    if bend > #slices then bend = #slices end
    output_gen[{{bstart, bend}}] = model:forward(input[{{bstart, bend}}])
    bstart = bend + 1
  end
  output_gen = preprocess_fn.deprocess(output_gen):float()
  output_gen:clamp(0,1)  -- in the network architecture, there is no guarentee that the output is in [0,1]
                         -- but it seems the network learnt that the value should be in this range 
  if plotSlice then
    local originalInput = preprocess_fn.deprocess(input)
    for i=1, output_gen:size(1) do
      local sliceId = tonumber(string.match(slices[i], ".*-(.*).png"))
      image.save(string.format("%s_%d_input.png", prefix, sliceId), originalInput[i])
      image.save(string.format("%s_%d_gen.png", prefix, sliceId), output_gen[i])
      image.save(string.format("%s_%d.png", prefix, sliceId), gtrueth[i])
    end
  end
  output_gen = output_gen:transpose(1,2)
  local mN, mH, mW = output_gen:max(2):squeeze(), output_gen:max(3):squeeze(), output_gen:max(4):squeeze()
  gtrueth = gtrueth:transpose(1,2)
  local mN_t, mH_t, mW_t = gtrueth:max(2):squeeze(), gtrueth:max(3):squeeze(), gtrueth:max(4):squeeze()
  image.save(string.format("%s_N_gen.png", prefix), mN)
  image.save(string.format("%s_H_gen.png", prefix), mH)
  image.save(string.format("%s_W_gen.png", prefix), mW)
  image.save(string.format("%s_N.png", prefix), mN_t)
  image.save(string.format("%s_H.png", prefix), mH_t)
  image.save(string.format("%s_W.png", prefix), mW_t)
end

local volumns = getVolumnMap()
local N = 1
local id = 0
local fileName = "/home/saxiao/tmp/mri/trainIdList.txt"
local file = io.open(fileName, 'a')
for volId, slices in pairs(volumns) do
  file:write(string.format("%s\n", volId))
  if not selectVolId or volId == selectVolId then 
    local prefix = string.format("%s/%s", outputdir, volId)
    transform(slices, prefix)
    id = id + 1
    print(id)
  end
  --if id == N then break end
end
file:close()
