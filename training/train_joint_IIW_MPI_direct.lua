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

--GPU 2
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
h6 = h5 - cudnn.SpatialConvolution(32, 3, 1, 1) - cudnn.Sigmoid()

model = nn.gModule({h0},{h6})
model = model:cuda()

criterion_mpi = nn.ParallelCriterion():add(nn.MSECriterion(),0.5)
criterion_mpi = criterion_mpi:cuda()

criterion_iiw = nn.ParallelCriterion():add(nn.WHDRHingeLossPara(0.12,0.08,1),1)
criterion_iiw = criterion_iiw:cuda()

criterion_iiw_test = nn.ParallelCriterion():add(nn.WHDRHingeLossPara(0.1,0.0,1),1)
criterion_iiw_test = criterion_iiw_test:cuda()

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



postfix = 'joint_IIW_MPI_direct'
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
savePath = '/mnt/codes/intrinsic/'
-- trainLogger = optim.Logger(paths.concat(savePath, string.format('train_%s.log',postfix)))

local file = '/mnt/codes/qingnan_codes/train_joint_IIW_MPI_direct.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
print('*******************train file*******************')
print(line)
print('*******************train file*******************')
-- trainLogger:add{'*******************train file*******************'}
-- trainLogger:add{line}
-- trainLogger:add{'*******************train file*******************'}

local trainSet = {}
local file = '/mnt/data/MPI_main_imageSplit-300-train.txt'
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local file = '/mnt/data/iiw_Learning_Lightness_train.txt'
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet

local testSet = {}
local file = '/mnt/data/MPI_main_imageSplit-fullsize-ChenSplit-test.txt'
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(testSet, line)
end
f:close()
local file = '/mnt/data/iiw_Learning_Lightness_test.txt'
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
  local current_loss_iiw = 0
  local current_testloss_iiw = 0
  local current_loss_mpi = 0
  local current_testloss_mpi = 0
  local count_iiw = 0
  local testcount_iiw = 0
  local count_mpi = 0
  local testcount_mpi = 0
  batch_size = batch_size or 4
  local order = torch.randperm(trainsetSize)
  local order_test = torch.randperm(testsetSize)

  for t = 1,trainsetSize,batch_size do
    iter = iter + 1
    local size = math.min(t + batch_size, trainsetSize + 1) - t

    local feval = function(x_new)
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

        local pred = model:forward(inputs)

        if string.match(inputFile, "iiw") then
          local labels = string.gsub(inputFile,'png','txt')
          local tempLoss =  criterion_iiw:forward({pred}, {labels})
          local grad = criterion_iiw:backward({pred}, {labels})
          model:backward(inputs, grad[1])
          loss = loss + tempLoss
          current_loss_iiw = current_loss_iiw + tempLoss
          count_iiw = count_iiw + 1
        elseif string.match(inputFile, "MPI") then
          local albedoFile = string.gsub(inputFile,'input','albedo')
          local albedo = torch.CudaTensor(1, 3, height, width)
          albedo[1] = image.load(albedoFile):cuda()
          local tempLoss =  criterion_mpi:forward({pred}, {albedo})
          local grad = criterion_mpi:backward({pred}, {albedo})
          model:backward(inputs, grad[1])
          loss = loss + tempLoss
          current_loss_mpi = current_loss_mpi + tempLoss
          count_mpi = count_mpi + 1
        end
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
    print(string.format('Iter: %d Current loss: %4f', iter, fs[1]))
    -- print(string.format('Iter: %d Current loss: %4f', iter, current_loss_iiw/count_iiw))

    if iter % 4 == 0 then
      local testloss= 0
      for i = 1,size do
        local inputFile = testSet[order_test[testCount]]
        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local inputs = torch.CudaTensor(1, 3, height, width)
        inputs[1] = tempInput
        inputs = inputs * 255

        local pred = model:forward(inputs)

        if string.match(inputFile, "iiw") then
          local labels = string.gsub(inputFile,'png','txt')
          local tempLoss =  criterion_iiw_test:forward({pred}, {labels})
          testloss = testloss + tempLoss
          current_testloss_iiw = current_testloss_iiw + tempLoss
          testcount_iiw = testcount_iiw + 1
        elseif string.match(inputFile, "MPI") then
          local albedoFile = string.gsub(inputFile,'clean','albedo')
          local albedo = torch.CudaTensor(1, 3, height, width)
          albedo[1] = image.load(albedoFile):cuda()
          local tempLoss =  criterion_mpi:forward({pred}, {albedo})
          testloss = testloss + tempLoss
          current_testloss_mpi = current_testloss_mpi + tempLoss
          testcount_mpi = testcount_mpi + 1
        end

        testCount = testCount + 1
        if testCount > testsetSize then
          testCount = 1
        end
      end
      testloss= testloss/size
      print(string.format('TestIter: %d Current loss: %4f', iter, testloss))
      -- trainLogger:add{string.format('TestIter: %d Current loss: %4f', iter, loss)}
    end
  end

  -- normalize loss
  return current_loss_iiw/count_iiw, current_testloss_iiw/testcount_iiw, current_loss_mpi/count_mpi, current_testloss_mpi/testcount_mpi
end

netfiles = '/mnt/codes/intrinsic/netfiles/'
timer = torch.Timer()
local bestTestLoss = 999999
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local train_loss_iiw,test_loss_iiw,train_loss_mpi,test_loss_mpi= step(batch_size,i)

    if bestTestLoss > test_loss_iiw then
      bestTestLoss = test_loss_iiw
      local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
      model:clearState()
      torch.save(filename, model)
      local filename = string.format('%sstate_%s_best.t7',netfiles,postfix)
      torch.save(filename, adam_state_save)
    end
    if i == 25 then
      epoch_judge = true
    end

    print(string.format('Epoch: %d Current iiw loss: %4f, mpi loss: %4f', i, train_loss_iiw, train_loss_mpi))
    -- trainLogger:add{string.format('Epoch: %d Current loss: %4f', i, loss)}
    print(string.format('Epoch: %d Current iiw test loss: %4f, mpi test loss: %4f', i, test_loss_iiw, test_loss_mpi))
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