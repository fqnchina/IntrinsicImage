local IIWscale, parent = torch.class('nn.IIWscale', 'nn.Module')

function IIWscale:__init(average)
  parent.__init(self)
  self.average = average or 0
end

function IIWscale:updateOutput(input)
  self.average = self.average or 0
  if self.average == 1 then
    r_value = input[{{},{1,3},{},{}}]
    input_origin = input[{{},{4,6},{},{}}]
    r_value = torch.sum(r_value,2)/3
  else
    r_value = input[{{},{1},{},{}}]
    input_origin = input[{{},{2,4},{},{}}]
  end
  -- print(r_value)
  local input_mean = torch.cmax(torch.sum(input_origin,2)/3,0.0000000001)
  r_value = torch.cmax(r_value,0.0000000001)

  self.output = torch.CudaTensor():resizeAs(input_origin)
  local reflection_var = torch.cdiv(r_value,input_mean)
  self.output[{{},{1},{},{}}] = torch.cmul(reflection_var,input_origin[{{},{1},{},{}}])
  self.output[{{},{2},{},{}}] = torch.cmul(reflection_var,input_origin[{{},{2},{},{}}])
  self.output[{{},{3},{},{}}] = torch.cmul(reflection_var,input_origin[{{},{3},{},{}}])

  return self.output
end

function IIWscale:updateGradInput(input, gradOutput)
  if self.average == 1 then
    r_value = input[{{},{1,3},{},{}}]
    input_origin = input[{{},{4,6},{},{}}]
    r_value = torch.sum(r_value)/3
  else
    r_value = input[{{},{1},{},{}}]
    input_origin = input[{{},{2,4},{},{}}]
  end

  local input_mean = torch.cmax(torch.sum(input_origin,2)/3,0.0000000001)

  self.gradInput = torch.CudaTensor():resizeAs(input):zero()
  if self.average == 1 then
    self.gradInput[{{},{1},{},{}}] = torch.cdiv(torch.sum(torch.cmul(gradOutput,input_origin),2),input_mean)/3
    self.gradInput[{{},{2},{},{}}] = torch.cdiv(torch.sum(torch.cmul(gradOutput,input_origin),2),input_mean)/3
    self.gradInput[{{},{3},{},{}}] = torch.cdiv(torch.sum(torch.cmul(gradOutput,input_origin),2),input_mean)/3
  else
    self.gradInput[{{},{1},{},{}}] = torch.cdiv(torch.sum(torch.cmul(gradOutput,input_origin),2),input_mean)
  end
  
  return self.gradInput
end