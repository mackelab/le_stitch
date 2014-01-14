function z = OffDiag(x)

z = x - diag(diag(x));

end