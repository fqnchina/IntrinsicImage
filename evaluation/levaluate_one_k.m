function error=levaluate_one_k(a,b)%a: input image, b: ground truth. Can not exchange the order.
	error=0;
	k=double(round(max(size(a))*0.1));
	tot=0;
	[m n d]=size(a);
	for i=1:round(k/2):m-round(k/2)
		for j=1:round(k/2):n-round(k/2)
			tot=tot+1;
			error=error+evaluate_one_k(a(i:min(m,i+k-1),j:min(n,j+k-1),:),b(i:min(m,i+k-1),j:min(n,j+k-1),:));
		end
	end
	newa=a;
	error=error/tot;
end
