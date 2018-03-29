inputDir = '/mnt/data/iiw-dataset/data/';
files = dir([inputDir '*.json']);
for x = 1:length(files)
	filename = [inputDir files(x).name];
	outfile = [inputDir files(x).name(1:end-5) '.txt']
	x

	fid=fopen(filename);
	tline = fgetl(fid);
	comp = {};
    points = {};
	count = 1;
	count_point = 1;
	while ischar(tline)
	    if strfind(tline,'"darker"')
	        temp = tline;
	        tline = fgetl(fid);
	        if strfind(tline,'"darker_method"')
	           comp(count).judge = temp(18:end-3);
	           if strcmp(temp(18:end-3),'E')
	               comp(count).judge = '0';
	           end
	           while true
	               tline = fgetl(fid);
	               if strfind(tline,'"darker_score"')
	                    comp(count).weight = tline(23:end-2);
	                    tline = fgetl(fid);
	                    tline = fgetl(fid);
	                    tline = fgetl(fid);
	                    comp(count).point1 = tline(17:end-2);
	                    tline = fgetl(fid);
	                    comp(count).point2 = tline(17:end);
	                    count = count + 1;
	                    break;
	               end
	           end
	        end
	    end
	    if strfind(tline,'"id"')
	        temp = tline;
	        tline = fgetl(fid);
	        if strfind(tline,'"min_separation"')
	            tline = fgetl(fid);
	            if strfind(tline,'"opaque"')
	                points(count_point).id = temp(13:end-2);
	                while true
	                    tline = fgetl(fid);
	                    if strfind(tline,'"x"')
	                        points(count_point).x = tline(12:end-2);
	                        tline = fgetl(fid);
	                        if strfind(tline,'"y"')
	                            points(count_point).y = tline(12:end-2);
	                        end
	                        count_point = count_point + 1;
	                        break;
	                    end
	                end
	            end
	        end
	    end
	    tline = fgetl(fid);
	end
	fclose(fid);

	fileID = fopen(outfile,'w');
	for n = 1:length(comp)
	    for m = 1:length(points)
	        if strcmp(comp(n).point1,points(m).id)
	            fprintf(fileID, sprintf('%s,%s,%s,%s',comp(n).weight,comp(n).judge,points(m).x,points(m).y));
	            break;
	        end
	    end
	    for m = 1:length(points)
	        if strcmp(comp(n).point2,points(m).id)
	            fprintf(fileID, sprintf(',%s,%s\n',points(m).x,points(m).y));
	            break;
	        end
	    end
	end
	fclose(fileID);
end

