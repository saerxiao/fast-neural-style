require 'gnuplot'

local function plotTrainVal()
  local modelId = "pixel"
  local metricFile = string.format("checkpoint/mri-%s/metrics.t7", modelId)
  local metrics = torch.load(metricFile)
  local trainLoss = metrics.train_loss_history
  local train_ts = torch.range(1, #trainLoss)
  local fileName = string.format("plot/%s-loss.png", modelId)
  gnuplot.pngfigure(fileName)
  gnuplot.plot({'train', train_ts, torch.Tensor(trainLoss), '+'}, {'validate', torch.Tensor(metrics.val_loss_history_ts), torch.Tensor(metrics.val_loss_history), '+'})
  gnuplot.raw("set yrange [0:500]")
  gnuplot.plotflush()
end

local function plotAllVal(key)
  local models = {"percept", "percept-notanh-c1", "percept-notanh-c2"}
  local metrics = {}
  for _, modelId in pairs(models) do
    local metricFileName = string.format("checkpoint/mri-%s/metrics.t7", modelId)
    local metricFile = torch.load(metricFileName)
    metrics[modelId] = metricFile
  end
  local fileName = "plot/val-percept-compareChannels.png"
  gnuplot.pngfigure(fileName)
  gnuplot.plot({models[1], torch.Tensor(metrics[models[1]].val_loss_history_ts), torch.Tensor(metrics[models[1]][key]), '+'},
               {models[2], torch.Tensor(metrics[models[2]].val_loss_history_ts), torch.Tensor(metrics[models[2]][key]), '+'},
               {models[3], torch.Tensor(metrics[models[3]].val_loss_history_ts), torch.Tensor(metrics[models[3]][key]), '+'}
               --{models[4], torch.Tensor(metrics[models[4]].val_loss_history_ts), torch.Tensor(metrics[models[4]][key]), '+'},
               --{models[5], torch.Tensor(metrics[models[5]].val_loss_history_ts), torch.Tensor(metrics[models[5]][key]), '+'}
               --{models[6], torch.Tensor(metrics[models[6]].val_loss_history_ts), torch.Tensor(metrics[models[6]][key]), '+'},
               --{models[7], torch.Tensor(metrics[models[7]].val_loss_history_ts), torch.Tensor(metrics[models[7]][key]), '+'}
              )
  --gnuplot.raw("set yrange [0:300]")
  gnuplot.plotflush()
end

local function plotGanMetrics(modelId, valkey, trainkey, validOnly)
  local metricFileName = string.format("checkpoint/mri-%s/metrics.t7", modelId)
  local metricFile = torch.load(metricFileName)
  local metricsVal = metricFile.val_loss_history
  local valsVal = torch.Tensor(#metricsVal)
  for i=1, #metricsVal do
    valsVal[i] = metricsVal[i][valkey]
  end
  local valsTrain
  if not validOnly then
    local metricsTrain = metricFile.train_loss_history
    valsTrain = torch.Tensor(#metricsTrain)
    for i=1, #metricsTrain do
      valsTrain[i] = metricsTrain[i][trainkey]
    end
  end
  local fileName = string.format("plot/%s-%s.png", modelId, trainkey)
  gnuplot.pngfigure(fileName)
  if validOnly then
    gnuplot.plot('val', torch.Tensor(metricFile.val_loss_history_ts), valsVal,'+')
  else
    gnuplot.plot({'train', torch.range(1,valsTrain:size(1)), valsTrain, '+'}, {'val', torch.Tensor(metricFile.val_loss_history_ts), valsVal,'+'})
  end
  --gnuplot.raw("set yrange [0,300]")
  gnuplot.plotflush()
end

--plotAllVal('val_loss_history', 'loss')
plotGanMetrics('percept-notanh-gan-burn100', 'loss_content', 'loss_content', true)




