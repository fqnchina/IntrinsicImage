require 'nn'
require 'optim'
require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'sys'
require 'nngraph'
require 'cudnn'
cudnn.fastest = true
cudnn.benchmark = true

--GPU 0
local function subnet()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet1()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet2()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 2, 2, 2, 2))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 2, 2, 2, 2))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet4()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 4, 4, 4, 4))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 4, 4, 4, 4))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet8()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 8, 8, 8, 8))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 8, 8, 8, 8))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet16()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 16, 16, 16, 16))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 16, 16, 16, 16))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet32()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 32, 32, 32, 32))
  sub:add(cudnn.SpatialBatchNormalization(32))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(32, 32, 3, 3, 1, 1, 32, 32, 32, 32))
  sub:add(cudnn.SpatialBatchNormalization(32))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

h0 = nn.Identity()()
h0_sub = h0 - nn.AddConstant(-115)

h1 = h0_sub - cudnn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true)
h2 = h1 - cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true)
h3 = h2 - cudnn.SpatialConvolution(32, 32, 3, 3, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true)
sub1 = h3 - subnet2() - nn.CAddTable() - cudnn.ReLU(true)
sub2 = sub1 - subnet2() - nn.CAddTable() - cudnn.ReLU(true)
sub3 = sub2 - subnet4() - nn.CAddTable() - cudnn.ReLU(true)
sub4 = sub3 - subnet4() - nn.CAddTable() - cudnn.ReLU(true)
sub5 = sub4 - subnet8() - nn.CAddTable() - cudnn.ReLU(true)
sub6 = sub5 - subnet8() - nn.CAddTable() - cudnn.ReLU(true)
sub7 = sub6 - subnet16() - nn.CAddTable() - cudnn.ReLU(true)
sub8 = sub7 - subnet16() - nn.CAddTable() - cudnn.ReLU(true)
sub9 = sub8 - subnet1() - nn.CAddTable() - cudnn.ReLU(true)
sub10 = sub9 - subnet1() - nn.CAddTable() - cudnn.ReLU(true)
h4 = sub10 - cudnn.SpatialFullConvolution(32, 32, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true)
h5 = h4 - cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(32) - cudnn.ReLU(true)
h6 = h5 - cudnn.SpatialConvolution(32, 1, 1, 1) - cudnn.Sigmoid()

model = nn.gModule({h0},{h6})
model = model:cuda()

criterion = nn.WHDRHingeLossPara(0.12,0.08,0)
criterion = criterion:cuda()

criterion_test = nn.WHDRHingeLossPara(0.1,0.0,0)
criterion_test = criterion_test:cuda()

model_edge = nn.EdgeComputation(100)

for i,module in ipairs(model:listModules()) do
   local m = module
   if m.__typename == 'cudnn.SpatialConvolution' or m.__typename == 'cudnn.SpatialFullConvolution' then
      local stdv = math.sqrt(12/(m.nInputPlane*m.kH*m.kW + m.nOutputPlane*m.kH*m.kW))
      m.weight:uniform(-stdv, stdv)
      m.bias:zero()
   end
   if m.__typename == 'cudnn.SpatialBatchNormalization' then
      m.weight:fill(1)
      m.bias:zero()
   end
end



postfix = 'IIW_direct'
max_iters = 35
batch_size = 1

model:training()
collectgarbage()

parameters, gradParameters = model:getParameters()

sgd_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-8,
  weightDecay = 0.0005,
  momentum = 0.9,
  dampening = 0,
  nesterov = true
}

adam_params = {
  learningRate = 1e-2,
  weightDecay = 0.0005,
  beta1 = 0.9,
  beta2 = 0.999
}

rmsprop_params = {
  learningRate = 1e-2,
  weightDecay = 0.0005,
  alpha = 0.9
}

-- Log results to files
savePath = '/raid/qingnan/codes/intrinsic/'

-- NOTE: the training and testing images have to be cropped to the size of even number to fit our network structure. 

local file = '/raid/qingnan/codes/qingnan_codes/train_IIW_direct.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
print('*******************train file*******************')
print(line)
print('*******************train file*******************')

local file = '/raid/qingnan/data/iiw_Learning_Lightness_train.txt'
local trainSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet

local file = '/raid/qingnan/data/iiw_Learning_Lightness_test.txt'
local testSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(testSet, line)
end
f:close()
local testsetSize = #testSet


local iter = 0
local epoch_judge = false
step = function(batch_size)
  local testCount = 1
  local current_loss= 0
  local current_testloss= 0
  local count = 0
  local testcount = 0
  batch_size = batch_size or 4
  local order = torch.randperm(trainsetSize)

  for t = 1,trainsetSize,batch_size do
    iter = iter + 1
    local size = math.min(t + batch_size, trainsetSize + 1) - t

    local feval = function(x_new)
      -- reset data
      if parameters ~= x_new then parameters:copy(x_new) end
      gradParameters:zero()

      local loss= 0
      for i = 1,size do
        local inputFile =  trainSet[order[t+i-1]]
        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local inputs = torch.CudaTensor(1, 3, height, width)
        inputs[1] = tempInput
        inputs = inputs * 255

        local labels = string.gsub(inputFile,'png','txt')

        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        local grad = criterion:backward(pred, labels)
        model:backward(inputs, grad)

        loss= loss+ tempLoss
      end
      gradParameters:div(size)
      loss= loss/size

      return loss, gradParameters
    end
    
    if epoch_judge then
      adam_params.learningRate = adam_params.learningRate*0.1
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params, adam_params)
      epoch_judge = false
    else
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params)
    end
    -- _, fs = optim.sgd(feval, parameters, sgd_params)

    count = count + 1
    current_loss= current_loss+ fs[1]
    print(string.format('Iter: %d Current loss: %4f', iter, fs[1]))
    -- trainLogger:add{string.format('Iter: %d Current loss: %4f', iter, fs[1])}

    if iter % 4 == 0 then
      local testloss= 0
      for i = 1,size do
        local inputFile = testSet[testCount]
        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local inputs = torch.CudaTensor(1, 3, height, width)
        inputs[1] = tempInput
        inputs = inputs * 255

        local labels = string.gsub(inputFile,'png','txt')

        local pred = model:forward(inputs)
        local temploss =  criterion_test:forward(pred, labels)
        testloss= testloss+ temploss

        testCount = testCount + 1
      end
      testloss= testloss/size
      testcount = testcount + 1
      current_testloss= current_testloss+ testloss

      print(string.format('TestIter: %d Current loss: %4f', iter, testloss))
      -- trainLogger:add{string.format('TestIter: %d Current loss: %4f', iter, loss)}
    end
  end

  -- normalize loss
  return current_loss/ count, current_testloss/ testcount
end

netfiles = '/raid/qingnan/codes/intrinsic/netfiles/'
timer = torch.Timer()
local bestTestLoss = 999999
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local train_loss,test_loss= step(batch_size,i)

    if i <= 25 then
      if bestTestLoss > test_loss then
        bestTestLoss = test_loss
        local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
        model:clearState()
        torch.save(filename, model)
      end
    else
      if bestTestLoss > test_loss then
        bestTestLoss = test_loss
        local filename = string.format('%smodel_%s_best_joint.net',netfiles,postfix)
        model:clearState()
        torch.save(filename, model)
      end
    end
    
    if i == 25 then
      epoch_judge = true

      local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
      model = torch.load(filename)
      model = model:cuda()
      model:training()

      parameters, gradParameters = model:getParameters()
      collectgarbage()
      
      bestTestLoss = 999999
    end


    print(string.format('Epoch: %d Current loss: %4f', i, train_loss))
    -- trainLogger:add{string.format('Epoch: %d Current loss: %4f', i, loss)}
    print(string.format('Epoch: %d Current test loss: %4f', i, test_loss))
    -- trainLogger:add{string.format('Epoch: %d Current test loss: %4f', i, testloss)}

    -- save/log current net
    local filename = string.format('%smodel_%s_%d.net',netfiles,postfix,i)
    model:clearState()
    torch.save(filename, model)
    local filename = string.format('%sstate_%s_%d.t7',netfiles,postfix,i)
    torch.save(filename, adam_state_save)
    print('Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours')
    -- trainLogger:add{'Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours'}
  end
end
print('Time elapsed: ' .. timer:time().real/(3600*24) .. ' days')
-- trainLogger:add{'Time elapsed: ' .. timer:time().real/(3600*24) .. ' days'}  
