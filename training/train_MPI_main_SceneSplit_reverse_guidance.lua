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

  sub:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())
  cont:add(cudnn.ReLU(true))

  return cont
end

local function subnet1()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet2()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 2, 2, 2, 2))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 2, 2, 2, 2))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet4()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 4, 4, 4, 4))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 4, 4, 4, 4))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet8()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 8, 8, 8, 8))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 8, 8, 8, 8))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet16()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 16, 16, 16, 16))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 16, 16, 16, 16))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet64()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 64, 64, 64, 64))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 64, 64, 64, 64))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

h0 = nn.Identity()()
h0_edge = h0 - nn.EdgeComputation(100)
h0_join = {h0,h0_edge} - nn.JoinTable(2) - nn.AddConstant(-115)

input_h0 = h0_join - cudnn.SpatialConvolution(4, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
input_h1 = input_h0 - cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
input_h2 = input_h1 - cudnn.SpatialConvolution(64, 64, 3, 3, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)

inter_h1 = input_h2 - subnet2() - nn.CAddTable() - cudnn.ReLU(true)
inter_h2 = inter_h1 - subnet2() - nn.CAddTable() - cudnn.ReLU(true)
inter_h3 = inter_h2 - subnet4() - nn.CAddTable() - cudnn.ReLU(true)
inter_h4 = inter_h3 - subnet4() - nn.CAddTable() - cudnn.ReLU(true)
inter_h5 = inter_h4 - subnet8() - nn.CAddTable() - cudnn.ReLU(true)
inter_h6 = inter_h5 - subnet8() - nn.CAddTable() - cudnn.ReLU(true)

output_h0 = inter_h6 - cudnn.SpatialFullConvolution(64, 64, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
output_h1 = output_h0 - cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
output_h2 = output_h1 - cudnn.SpatialConvolution(64, 1, 1, 1)
output_h3 = {output_h2,h0_edge} - nn.CAddTable()

model = nn.gModule({h0},{output_h3})
model = model:cuda()

criterion = nn.MSECriterion()
criterion = criterion:cuda()

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



postfix = 'MPI_main_SceneSplit_reverse_guidance'
max_iters = 40
batch_size = 16

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

local file = '/raid/qingnan/codes/qingnan_codes/train_MPI_main_SceneSplit_reverse_guidance.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
print('*******************train file*******************')
print(line)
print('*******************train file*******************')

local file = '/raid/qingnan/data/MPI_main_sceneSplit-300-test.txt'
local trainSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet

local file = '/raid/qingnan/data/MPI_main_sceneSplit-fullsize-NoDefect-train.txt'
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

        local albedoFile = string.gsub(inputFile,'input','albedo')
        local tempInput = image.load(inputFile):cuda()
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local albedo = torch.CudaTensor(1, 3, height, width)
        input[1] = image.load(inputFile):cuda()
        albedo[1] = image.load(albedoFile):cuda()
        input = input * 255
        albedo = albedo * 255
        local label = model_edge:forward(albedo)
        
        local pred = model:forward(input)
        local tempLoss =  criterion:forward(pred, label)
        local grad = criterion:backward(pred, label)
        model:backward(input, grad)

        loss = loss + tempLoss
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

    count = count + 1
    current_loss= current_loss+ fs[1]
    print(string.format('Iter: %d Current loss: %4f', iter, fs[1]))

    if iter % 10 == 0 then
      local testloss= 0
      for i = 1,size do
        local inputFile = testSet[testCount]

        local albedoFile = string.gsub(inputFile,'clean','albedo')
        local tempInput = image.load(inputFile):cuda()
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local albedo = torch.CudaTensor(1, 3, height, width)
        input[1] = image.load(inputFile):cuda()
        albedo[1] = image.load(albedoFile):cuda()
        input = input * 255
        albedo = albedo * 255
        local label = model_edge:forward(albedo)

        local pred = model:forward(input)
        local tempLoss =  criterion:forward(pred, label)
        testloss = testloss + tempLoss

        testCount = testCount + 1
        if testCount >  testsetSize then
          testCount = 1
        end
      end
      testloss= testloss/size
      testcount = testcount + 1
      current_testloss= current_testloss+ testloss

      print(string.format('TestIter: %d Current loss: %4f', iter, testloss))
    end
  end

  return current_loss/ count, current_testloss/ testcount
end

netfiles = '/raid/qingnan/codes/intrinsic/netfiles/'
timer = torch.Timer()
local bestTestLoss = 999999
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local train_loss,test_loss= step(batch_size,i)

    if i <= 35 then
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
    
    if i == 35 then
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
    print(string.format('Epoch: %d Current test loss: %4f', i, test_loss))

    -- save/log current net
    local filename = string.format('%smodel_%s_%d.net',netfiles,postfix,i)
    model:clearState()
    torch.save(filename, model)
    local filename = string.format('%sstate_%s_%d.t7',netfiles,postfix,i)
    torch.save(filename, adam_state_save)
    print('Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours')
  end
end
print('Time elapsed: ' .. timer:time().real/(3600*24) .. ' days')
