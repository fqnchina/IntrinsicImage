local WHDRHingeLossPara, parent = torch.class('nn.WHDRHingeLossPara', 'nn.Criterion')

function WHDRHingeLossPara:__init(delta, epsilon, average)
   parent.__init(self)
   self.delta = delta or 0.1
   self.epsilon = epsilon or 0
   self.average = average or 0
end

function WHDRHingeLossPara:updateOutput(input, targetFile)
   local bs,dim,height,width = input:size(1),input:size(2),input:size(3),input:size(4)
   if self.average == 1 then
      input = torch.cmax(torch.sum(input,2)/3,0.0000000001)
   else
      input = torch.cmax(input,0.0000000001)
   end
   
   -- self.output = 0
   self.whdr = 0
   self.weight = 0
   for line in io.lines(targetFile) do 
      strs = line:split(",")
      self.weight = self.weight + tonumber(strs[1])
      local point1_x = math.floor(width*tonumber(strs[3]))
      local point1_y = math.floor(height*tonumber(strs[4]))
      local point2_x = math.floor(width*tonumber(strs[5]))
      local point2_y = math.floor(height*tonumber(strs[6]))
      local ratio = input[1][1][point1_y][point1_x]/input[1][1][point2_y][point2_x]

      local predict_j = -1
      if ratio > (1+self.delta) then
         predict_j = 2
      elseif ratio < 1/(1+self.delta) then
         predict_j = 1
      else
         predict_j = 0         
      end
      if tonumber(strs[2]) ~= predict_j then
         self.whdr = self.whdr + tonumber(strs[1])
      end

      -- if tonumber(strs[2]) == 0 then
      --    if ratio < 0.95 then
      --       self.output = self.output + (0.95 - ratio) * tonumber(strs[1])
      --    elseif  ratio > 1.05 then
      --       self.output = self.output + (ratio - 1.05) * tonumber(strs[1])
      --    end
      -- elseif tonumber(strs[2]) == 1 then
      --    if ratio > 0.87 then
      --       self.output = self.output + (ratio - 0.87) * tonumber(strs[1])
      --    end
      -- else
      --    if ratio < 1.15 then
      --       self.output = self.output + (1.15 - ratio) * tonumber(strs[1])
      --    end
      -- end
   end

   -- self.output = self.output / self.weight
   self.whdr = self.whdr / self.weight
   return self.whdr
end

function WHDRHingeLossPara:updateGradInput(input, targetFile)
   local height,width = input:size(3),input:size(4)
   self.gradInput = torch.CudaTensor():resizeAs(input):zero()

   if self.average == 1 then
      input = torch.cmax(torch.sum(input,2)/3,0.0000000001)
   else
      input = torch.cmax(input,0.0000000001)
   end

   for line in io.lines(targetFile) do 
      local strs = line:split(",")
      local point1_x = math.floor(width*tonumber(strs[3]))
      local point1_y = math.floor(height*tonumber(strs[4]))
      local point2_x = math.floor(width*tonumber(strs[5]))
      local point2_y = math.floor(height*tonumber(strs[6]))
      local ratio = input[1][1][point1_y][point1_x]/input[1][1][point2_y][point2_x]
      
      if tonumber(strs[2]) == 0 then
         if ratio < 1/(1+self.delta-self.epsilon) then
            self.gradInput[{{},{},{point1_y},{point1_x}}] = self.gradInput[{{},{},{point1_y},{point1_x}}] - 1/input[1][1][point2_y][point2_x] * tonumber(strs[1])
            self.gradInput[{{},{},{point2_y},{point2_x}}] = self.gradInput[{{},{},{point2_y},{point2_x}}] + input[1][1][point1_y][point1_x]/(input[1][1][point2_y][point2_x]*input[1][1][point2_y][point2_x]) * tonumber(strs[1])
         elseif  ratio > (1+self.delta-self.epsilon) then
            self.gradInput[{{},{},{point1_y},{point1_x}}] = self.gradInput[{{},{},{point1_y},{point1_x}}] + 1/input[1][1][point2_y][point2_x] * tonumber(strs[1])
            self.gradInput[{{},{},{point2_y},{point2_x}}] = self.gradInput[{{},{},{point2_y},{point2_x}}] - input[1][1][point1_y][point1_x]/(input[1][1][point2_y][point2_x]*input[1][1][point2_y][point2_x]) * tonumber(strs[1])
         end
      elseif tonumber(strs[2]) == 1 then
         if ratio > 1/(1+self.delta+self.epsilon) then
            self.gradInput[{{},{},{point1_y},{point1_x}}] = self.gradInput[{{},{},{point1_y},{point1_x}}] + 1/input[1][1][point2_y][point2_x] * tonumber(strs[1])
            self.gradInput[{{},{},{point2_y},{point2_x}}] = self.gradInput[{{},{},{point2_y},{point2_x}}] - input[1][1][point1_y][point1_x]/(input[1][1][point2_y][point2_x]*input[1][1][point2_y][point2_x]) * tonumber(strs[1])
         end
      else
         if ratio < (1+self.delta+self.epsilon) then
            self.gradInput[{{},{},{point1_y},{point1_x}}] = self.gradInput[{{},{},{point1_y},{point1_x}}] - 1/input[1][1][point2_y][point2_x] * tonumber(strs[1])
            self.gradInput[{{},{},{point2_y},{point2_x}}] = self.gradInput[{{},{},{point2_y},{point2_x}}] + input[1][1][point1_y][point1_x]/(input[1][1][point2_y][point2_x]*input[1][1][point2_y][point2_x]) * tonumber(strs[1])
         end
      end
   end

   self.gradInput = self.gradInput / self.weight
   if self.average == 1 then
      self.gradInput = self.gradInput / 3
   end

   return self.gradInput 
end
