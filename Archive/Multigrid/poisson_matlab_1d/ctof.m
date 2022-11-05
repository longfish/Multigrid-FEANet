function uf = ctof ( nc, uc, nf, uf )

%*****************************************************************************80
%                                                    
%% ctof() transfers data from a coarse to a finer grid.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    06 December 2011
%
%  Author:
%
%    John Burkardt
%
%  Reference:
%
%    William Hager,
%    Applied Numerical Linear Algebra,
%    Prentice-Hall, 1988,
%    ISBN13: 978-0130412942,
%    LC: QA184.H33.
%
%  Input:
%
%    integer NC, the number of coarse nodes.
%
%    real UC(NC,1), the coarse correction data.
%
%    integer NF, the number of fine nodes.
%
%    real UF(NF,1), the fine grid data.
%
%  Output:
%
%    real UF(NF,1), the data has been updated with prolonged coarse 
%    correction data.
% 
  uf(1:2:2*nc-1,1) = uf(1:2:2*nc-1,1) + uc(1:nc,1);
  uf(2:2:2*nc-2,1) = uf(2:2:2*nc-2,1) + 0.5 * ( uc(1:nc-1,1) + uc(2:nc,1) );

  return
end
