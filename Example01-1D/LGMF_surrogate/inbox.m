function in = inbox(testpts,xyz)
% arguments: (input)
%  testpts - nxp array to test, n data points, in p dimensions
%
%  xyz - mxp array of points about which to construct the box
%
% arguments: (output)
%  in  - nx1 logical vector
%        in(i) == 1 --> the i'th point was inside the box.

max_dims = max(xyz);
min_dims = min(xyz);

n = size(testpts, 1);
max_dims = repmat(max_dims, [n, 1]);
min_dims = repmat(min_dims, [n, 1]);

dims_in = testpts <= max_dims & testpts >= min_dims;
in = all(dims_in, 2);