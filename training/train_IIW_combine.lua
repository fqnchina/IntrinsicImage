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

--GPU 1
model_guidance = torch.load('/raid/qingnan/codes/intrinsic/netfiles/model_IIW_guidance_best_joint.net')
model_intrinsic = torch.load('/raid/qingnan/codes/intrinsic/netfiles/model_IIW_direct_best_joint.net')

h0 = nn.Identity()()
r = h0 - model_intrinsic
g = h0 - model_guidance
R = {r,h0} - nn.JoinTable(2) - nn.IIWscale()
R_final = {R,g} - nn.JoinTable(2) - nn.DomainTransform(5,0.25,30,0)

model = nn.gModule({h0},{r,g,R_final})
model = model:cuda()

criterion = nn.ParallelCriterion():add(nn.WHDRHingeLossPara(0.12,0.08,0),1):add(nn.MSECriterion(),0.35):add(nn.WHDRHingeLossPara(0.12,0.08,1),0.1)
criterion = criterion:cuda()

criterion_test1 = nn.WHDRHingeLossPara(0.1,0,0)
criterion_test3 = nn.WHDRHingeLossPara(0.1,0,1)
criterion_test2 = nn.MSECriterion()
criterion_test1 = criterion_test1:cuda()
criterion_test3 = criterion_test3:cuda()
criterion_test2 = criterion_test2:cuda()

model_edge = nn.EdgeComputation(100)




postfix = 'IIW_comnbine'
max_iters = 5
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
  learningRate = 0.001,
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
-- trainLogger = optim.Logger(paths.concat(savePath, string.format('train_%s.log',postfix)))

local file = '/raid/qingnan/codes/qingnan_codes/smooth_train.lua'
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
  local current_loss1 = 0
  local current_loss2 = 0
  local current_loss3 = 0
  local current_testloss1 = 0
  local current_testloss2 = 0
  local current_testloss3 = 0
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

      local loss1 = 0
      local loss2 = 0
      local loss3 = 0
      for i = 1,size do
        local inputFile =  trainSet[order[t+i-1]]
        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local inputs = torch.CudaTensor(1, 3, height, width)
        inputs[1] = tempInput
        inputs = inputs * 255

        local labelFile = string.gsub(inputFile,'iiw%-dataset/data_even','iiw_L1_Flattening')
        local label = torch.CudaTensor(1, 3, height, width)
        label[1] = image.load(labelFile)
        label = label * 255

        local label1 = string.gsub(inputFile,'png','txt')
        local label2 = model_edge:forward(label)
        labels = {label1,label2,label1}

        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        local grad = criterion:backward(pred, labels)
        model:backward(inputs, grad)

        local pred1 = pred[1]
        local temploss =  criterion_test1:forward(pred1, label1)
        loss1 = loss1 + temploss
        local pred3 = pred[3]
        local temploss =  criterion_test3:forward(pred3, label1)
        loss3 = loss3 + temploss
        local pred2 = pred[2]
        local temploss =  criterion_test2:forward(pred2, label2)
        loss2 = loss2 + temploss
      end
      gradParameters:div(size)
      loss1 = loss1/size
      loss2 = loss2/size
      loss3 = loss3/size

      return {loss1,loss2,loss3}, gradParameters
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
    current_loss1 = current_loss1 + fs[1][1]
    current_loss2 = current_loss2 + fs[1][2]
    current_loss3 = current_loss3 + fs[1][3]
    print(string.format('Iter: %d Current loss1: %4f, loss3: %4f, loss2: %4f', iter, fs[1][1], fs[1][3], fs[1][2]))
    -- trainLogger:add{string.format('Iter: %d Current loss: %4f', iter, fs[1])}

    if iter % 4 == 0 then
      local testloss1 = 0
      local testloss2 = 0
      local testloss3 = 0
      for i = 1,size do
        local inputFile = testSet[testCount]
        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local inputs = torch.CudaTensor(1, 3, height, width)
        inputs[1] = tempInput
        inputs = inputs * 255

        local labelFile = string.gsub(inputFile,'iiw%-dataset/data_even','iiw_L1_Flattening')
        local label = torch.CudaTensor(1, 3, height, width)
        label[1] = image.load(labelFile)
        label = label * 255
        local label1 = string.gsub(inputFile,'png','txt')
        local label2 = model_edge:forward(label)

        local pred = model:forward(inputs)

        local pred1 = pred[1]
        local temploss =  criterion_test1:forward(pred1, label1)
        testloss1 = testloss1 + temploss
        local pred3 = pred[3]
        local temploss =  criterion_test3:forward(pred3, label1)
        testloss3 = testloss3 + temploss
        local pred2 = pred[2]
        local temploss =  criterion_test2:forward(pred2, label2)
        testloss2 = testloss2 + temploss

        testCount = testCount + 1
      end
      testloss1 = testloss1/size
      testloss2 = testloss2/size
      testloss3 = testloss3/size
      testcount = testcount + 1
      current_testloss1 = current_testloss1 + testloss1
      current_testloss2 = current_testloss2 + testloss2
      current_testloss3 = current_testloss3 + testloss3

      print(string.format('TestIter: %d Current loss1: %4f, loss3: %4f, loss2: %4f', iter, testloss1, testloss3, testloss2))
      -- trainLogger:add{string.format('TestIter: %d Current loss: %4f', iter, loss)}
    end
  end

  -- normalize loss
  return current_loss1 / count, current_loss2 / count, current_loss3 / count, current_testloss1 / testcount, current_testloss3 / testcount, current_testloss2 / testcount
end

netfiles = '/raid/qingnan/codes/intrinsic/netfiles/'
timer = torch.Timer()
local bestTestLoss = 999999
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local train_loss1,train_loss2,train_loss3,test_loss1,test_loss3,test_loss2 = step(batch_size,i)
    if i <= 20 then
      if bestTestLoss > test_loss3 then
        bestTestLoss = test_loss3
        local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
        model:clearState()
        torch.save(filename, model)
      end
    else
      if bestTestLoss > test_loss3 then
        bestTestLoss = test_loss3
        local filename = string.format('%smodel_%s_best_joint.net',netfiles,postfix)
        model:clearState()
        torch.save(filename, model)
      end
    end
    
    if i == 20 then
      epoch_judge = true
      criterion = nn.ParallelCriterion():add(nn.WHDRHingeLoss(0),1):add(nn.MSECriterion(),0.35):add(nn.WHDRHingeLoss(1),0.1)
      criterion = criterion:cuda()

      local filename = string.format('%smodel_%s_best.net',netfiles,postfix)
      model = torch.load(filename)
      model = model:cuda()
      model:training()

      parameters, gradParameters = model:getParameters()
      collectgarbage()
      
      bestTestLoss = 999999
    end

    print(string.format('Epoch: %d Current loss1: %4f, loss3: %4f, loss2: %4f', i, train_loss1, train_loss3, train_loss2))
    -- trainLogger:add{string.format('Epoch: %d Current loss: %4f', i, loss)}
    print(string.format('Epoch: %d Current test loss1: %4f, loss3: %4f, loss2: %4f', i, test_loss1, test_loss3, test_loss2))
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
