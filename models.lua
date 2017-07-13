require 'nn'
require 'nngraph'

local model = {}

local function convModule(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, depth)
  if not depth then depth = 2 end
  local convInput = input
  if torch.type(input) == "table" then
    convInput = input - nn.JoinTable(2)
  end
  local c = convInput
  local n1, n2 = nInputPlane, nOutputPlane
  for i = 1, depth do
    c = c
        - nn.SpatialConvolutionMM(n1, n2,kW,kH,dW,dH,padW,padH)
        - nn.SpatialBatchNormalization(nOutputPlane)
        - nn.ReLU()
    n1 = n2
  end
  return c
end

local function convModule1(input, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local convInput = input
  if torch.type(input) == "table" then
    convInput = input - nn.JoinTable(2)
  end
  local c = convInput
            - nn.SpatialConvolutionMM(nInputPlane, nOutputPlane,kW,kH,dW,dH,padW,padH)
            - nn.SpatialBatchNormalization(nOutputPlane)
            - nn.ReLU()
  return c
end

function model.uNet()
  local input = - nn.Identity()
  -- contracting path
  local c1 = convModule(input,3,32,3,3,1,1,1,1)  -- receptive field: (1+2+2) x (1+2+2)
  local pool1 = c1 - nn.SpatialMaxPooling(2,2)   -- receptive field: 10x10
  local c2 = convModule(pool1,32,64,3,3,1,1,1,1) -- receptive field: 14x14
  local pool2 = c2 - nn.SpatialMaxPooling(2,2)   -- receptive field: 28x28
  local c3 = convModule(pool2,64,128,3,3,1,1,1,1) -- 32x32
  local pool3 = c3 - nn.SpatialMaxPooling(2,2)   -- 64x64
  local c4 = convModule(pool3,128,256,3,3,1,1,1,1) -- 68x68
  local pool4 = c4 - nn.SpatialMaxPooling(2,2)   -- 136x136
  local c5 = convModule(pool4,256,512,3,3,1,1,1,1) -- 140x140

  -- expansive path
  local up1 = c5 - nn.SpatialUpSamplingNearest(2)
  local c4Mirror = convModule({up1,c4},512+256,256,3,3,1,1,1,1)
  local up2 = c4Mirror - nn.SpatialUpSamplingNearest(2)
  local c3Mirror = convModule({up2,c3},256+128,128,3,3,1,1,1,1)
  local up3 = c3Mirror - nn.SpatialUpSamplingNearest(2)
  local c2Mirror = convModule({up3,c2},128+64,64,3,3,1,1,1,1)
  local up4 = c2Mirror - nn.SpatialUpSamplingNearest(2)
  local c1Mirror = convModule({up4,c1},64+32,32,3,3,1,1,1,1)

  -- make the right shape as the input
  local last = c1Mirror
               - nn.SpatialConvolutionMM(32,3,1,1,1,1,0,0)
  local g = nn.gModule({input},{last})
  return g
end

return model
