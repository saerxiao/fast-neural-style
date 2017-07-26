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

local function plotAllVal()
  local models = {"percept", "percept-notanh-c1", "percept-notanh-c2", "percept-notanh-c3", "percept-notanh-c23"}
  --local models = {"percept-notanh-2-4"}
  local metrics = {}
  for _, modelId in pairs(models) do
    local metricFileName = string.format("checkpoint/mri-%s/metrics.t7", modelId)
    local metricFile = torch.load(metricFileName)
    metrics[modelId] = metricFile
  end
  local fileName = "plot/val-percept-compareChannels.png"
  gnuplot.pngfigure(fileName)
  gnuplot.plot({models[1], torch.Tensor(metrics[models[1]].val_loss_history_ts), torch.Tensor(metrics[models[1]].val_loss_history), '+'},
               {models[2], torch.Tensor(metrics[models[2]].val_loss_history_ts), torch.Tensor(metrics[models[2]].val_loss_history), '+'},
               {models[3], torch.Tensor(metrics[models[3]].val_loss_history_ts), torch.Tensor(metrics[models[3]].val_loss_history), '+'},
               {models[4], torch.Tensor(metrics[models[4]].val_loss_history_ts), torch.Tensor(metrics[models[4]].val_loss_history), '+'},
               {models[5], torch.Tensor(metrics[models[5]].val_loss_history_ts), torch.Tensor(metrics[models[5]].val_loss_history), '+'}
               --{models[6], torch.Tensor(metrics[models[6]].val_loss_history_ts), torch.Tensor(metrics[models[6]].val_loss_history), '+'},
               --{models[7], torch.Tensor(metrics[models[7]].val_loss_history_ts), torch.Tensor(metrics[models[7]].val_loss_history), '+'}
              )
  --gnuplot.raw("set yrange [0:300]")
  gnuplot.plotflush()
end

plotAllVal()




