function error=evaluate_ssim_one_k(a,b)%a: input image, b: ground truth. Can not exchange the order.
	[m n d]=size(a);
	a(isnan(a))=0;
	va=reshape(a,[],1);
	vb=reshape(b,[],1);
	k=(va'*vb)/(va'*va);
	for i=2:10
		for j=-1:2:1
			step=j*2^-i;
			while(k+step<5&&k+step>0&&compute(a,b,k+step)>compute(a,b,k))
				k=k+step;
			end
		end
	end
	[error newa]=compute(a,b,k);
end
function [error newa]=compute(a,b,k)
	error=0;
	newa=a*k;
	for i=1:3
		error=error+ssim_index(newa(:,:,i)*255,b(:,:,i)*255)/3;
	end
end
